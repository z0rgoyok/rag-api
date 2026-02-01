"""Core agentic RAG implementation using ReAct pattern."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from core.config import Settings
from core.db import Db
from core.embeddings_client import EmbeddingsClient

from .protocol import AgentResult, AgentState, SearchResult, ToolName
from .tools import FinalAnswerTool, RefineAndSearchTool, SearchTool, get_tools_schema

log = logging.getLogger("rag_agent")


SYSTEM_PROMPT = """You are a research assistant with access to a knowledge base.
Your task is to answer the user's question using the available tools.

Strategy:
1. Start by searching the knowledge base with a query derived from the user's question.
2. Analyze the search results. If they don't fully answer the question, use refine_and_search with a different query angle.
3. You can perform up to {max_iterations} search iterations total.
4. Once you have enough information (or exhausted search options), use final_answer to respond.

Important:
- Base your answer ONLY on information from search results.
- If the knowledge base doesn't contain relevant information, say so honestly.
- Cite sources when providing information.
- Be concise but complete."""


@dataclass
class AgentConfig:
    """Configuration for the agent."""

    max_iterations: int = 3
    top_k: int = 6
    include_sources: bool = True


class Agent:
    """ReAct-style agent for knowledge base Q&A."""

    def __init__(
        self,
        *,
        db: Db,
        embed_client: EmbeddingsClient,
        chat_client: Any,  # ChatClient protocol
        settings: Settings,
        config: AgentConfig | None = None,
    ) -> None:
        self.db = db
        self.embed_client = embed_client
        self.chat_client = chat_client
        self.settings = settings
        self.config = config or AgentConfig()

        # Initialize tools
        self.search_tool = SearchTool(
            db=db,
            embed_client=embed_client,
            embeddings_model=settings.embeddings_model,
            top_k=self.config.top_k,
        )
        self.refine_tool = RefineAndSearchTool(
            db=db,
            embed_client=embed_client,
            embeddings_model=settings.embeddings_model,
            top_k=self.config.top_k,
        )
        self.final_answer_tool = FinalAnswerTool()

        self.tools_by_name = {
            "search": self.search_tool,
            "refine_and_search": self.refine_tool,
            "final_answer": self.final_answer_tool,
        }

    async def run(self, query: str) -> AgentResult:
        """Run the agent loop to answer a query."""
        state = AgentState(original_query=query)

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(max_iterations=self.config.max_iterations),
            },
            {"role": "user", "content": query},
        ]

        while state.iterations < self.config.max_iterations:
            state.iterations += 1
            log.info("agent_iteration=%d query=%s", state.iterations, query[:100])

            # Call LLM with tools
            payload = {
                "model": self.settings.chat_model,
                "messages": messages,
                "tools": get_tools_schema(),
                "tool_choice": "auto",
            }

            try:
                response = await self.chat_client.chat_completions(payload)
            except Exception as e:
                log.error("agent_llm_error=%s", e)
                return self._build_fallback_result(state, f"LLM error: {e}")

            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            finish_reason = choice.get("finish_reason")

            # Add assistant message to history
            messages.append(message)

            # Check for tool calls
            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                # No tool calls - check if there's a direct response
                content = message.get("content", "")
                if content and finish_reason == "stop":
                    state.final_answer = content
                    break
                # LLM didn't call tools and didn't provide answer - force final answer
                log.warning("agent_no_tools_no_answer iteration=%d", state.iterations)
                break

            # Execute tool calls
            for tc in tool_calls:
                tool_name = tc.get("function", {}).get("name", "")
                tool_args_str = tc.get("function", {}).get("arguments", "{}")
                tool_call_id = tc.get("id", "")

                try:
                    tool_args = json.loads(tool_args_str)
                except json.JSONDecodeError:
                    tool_args = {}

                log.info("agent_tool_call tool=%s args=%s", tool_name, tool_args)
                state.add_reasoning(f"Calling {tool_name}: {tool_args}")

                tool = self.tools_by_name.get(tool_name)
                if tool is None:
                    result_content = json.dumps({"error": f"Unknown tool: {tool_name}"})
                else:
                    result = await tool.execute(tool_args, state)
                    if result.tool_name == ToolName.FINAL_ANSWER and result.success:
                        # Agent provided final answer
                        break
                    result_content = self._format_tool_result(result)

                # Add tool result to messages
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": result_content,
                    }
                )

            # Check if final answer was provided
            if state.final_answer is not None:
                break

        # Build result
        return self._build_result(state)

    def _format_tool_result(self, result: Any) -> str:
        """Format tool result for the LLM."""
        if not result.success:
            return json.dumps({"error": result.error})

        if result.tool_name in (ToolName.SEARCH, ToolName.REFINE_AND_SEARCH):
            # Format search results
            results: list[SearchResult] = result.data
            if not results:
                return json.dumps({"results": [], "message": "No relevant results found."})

            formatted = []
            for r in results:
                entry = {
                    "content": r.content[:1000],  # Truncate for LLM context
                    "source": r.source,
                    "score": round(r.score, 3),
                }
                if r.page is not None:
                    entry["page"] = r.page
                formatted.append(entry)

            return json.dumps({"results": formatted, "count": len(formatted)})

        return json.dumps({"data": result.data})

    def _build_result(self, state: AgentState) -> AgentResult:
        """Build the final agent result."""
        all_results = state.get_all_results()

        sources = []
        if self.config.include_sources:
            for r in all_results[:10]:  # Limit sources
                sources.append(
                    {
                        "title": r.source.split("/")[-1],
                        "path": r.source,
                        "page": r.page,
                        "score": r.score,
                    }
                )

        answer = state.final_answer
        if not answer:
            # Fallback: synthesize from search results if no explicit answer
            if all_results:
                answer = "Based on the search results, I found relevant information but couldn't formulate a complete answer. Please try rephrasing your question."
            else:
                answer = "I couldn't find relevant information in the knowledge base to answer your question."

        return AgentResult(
            answer=answer,
            sources=sources,
            reasoning_steps=state.reasoning_steps,
            search_count=len(state.search_history),
            iterations=state.iterations,
        )

    def _build_fallback_result(self, state: AgentState, error: str) -> AgentResult:
        """Build a fallback result when the agent fails."""
        return AgentResult(
            answer=f"I encountered an error while processing your question: {error}",
            sources=[],
            reasoning_steps=state.reasoning_steps,
            search_count=len(state.search_history),
            iterations=state.iterations,
        )

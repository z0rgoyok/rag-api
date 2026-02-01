# Agentic RAG

Модуль агентского RAG с итеративным поиском и reasoning-ом. Использует паттерн ReAct (Reasoning + Acting) для многошагового ответа на сложные вопросы.

## Когда использовать

| Сценарий | Обычный RAG | Агентский RAG |
|----------|-------------|---------------|
| Простой фактологический вопрос | ✅ | Избыточно |
| Вопрос требует синтеза из нескольких источников | ❌ | ✅ |
| Сравнительный анализ | ❌ | ✅ |
| Неоднозначный запрос | ❌ | ✅ |
| Критична низкая латентность | ✅ | ❌ |

## API

### Endpoint

```
POST /v1/agent/chat
```

### Request

```json
{
  "query": "Сравни подходы к обработке ошибок в книгах A и B",
  "max_iterations": 3,
  "citations": true
}
```

| Поле | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `query` | string | required | Вопрос пользователя |
| `max_iterations` | int | 3 | Макс. итераций агента (1-10) |
| `citations` | bool | false | Включить источники в ответ |

### Response

```json
{
  "answer": "В книге A используется подход X, тогда как в книге B предпочитают Y...",
  "sources": [
    {"title": "book_a.pdf", "path": "/data/book_a.pdf", "page": 42, "score": 0.89},
    {"title": "book_b.pdf", "path": "/data/book_b.pdf", "page": 15, "score": 0.85}
  ],
  "reasoning_steps": [
    "Calling search: {\"query\": \"error handling approaches\"}",
    "Refining search with: error handling book A",
    "Calling refine_and_search: {\"refined_query\": \"error handling book B comparison\"}"
  ],
  "search_count": 3,
  "iterations": 2
}
```

| Поле | Тип | Описание |
|------|-----|----------|
| `answer` | string | Финальный ответ агента |
| `sources` | array | Источники (если `citations=true` и разрешено тарифом) |
| `reasoning_steps` | array | Шаги рассуждения агента |
| `search_count` | int | Количество выполненных поисков |
| `iterations` | int | Количество итераций агента |

### Примеры использования

**curl:**
```bash
curl -X POST http://localhost:18080/v1/agent/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "query": "What are the main differences between microservices and monolithic architecture?",
    "max_iterations": 3,
    "citations": true
  }'
```

**Python:**
```python
import httpx

response = httpx.post(
    "http://localhost:18080/v1/agent/chat",
    json={
        "query": "Какие паттерны проектирования описаны в книге?",
        "max_iterations": 3,
        "citations": True,
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"},
)
result = response.json()
print(result["answer"])
for step in result["reasoning_steps"]:
    print(f"  - {step}")
```

---

## Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                         Agent Loop                               │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Query   │───▶│   LLM    │───▶│  Tools   │───▶│  State   │  │
│  └──────────┘    │ (decide) │    │ (execute)│    │ (update) │  │
│                  └────┬─────┘    └──────────┘    └────┬─────┘  │
│                       │                               │         │
│                       └───────────────────────────────┘         │
│                              (loop until done)                  │
└─────────────────────────────────────────────────────────────────┘
```

### Компоненты

```
apps/agent/
├── __init__.py      # Package marker
├── protocol.py      # Типы и протоколы
├── tools.py         # Реализация инструментов
├── agent.py         # Основной агент (ReAct loop)
└── README.md        # Эта документация
```

### Зависимости

```
apps/agent/
    ├── core/config.py          # Settings
    ├── core/db.py              # Database access
    ├── core/embeddings_client.py  # Embeddings
    └── core/pgvector.py        # Vector operations
```

Модуль `apps/agent/` зависит только от `core/`, не от других apps (согласно архитектуре проекта).

---

## Код: protocol.py

Определяет типы данных и протоколы для агентской системы.

### ToolName

```python
class ToolName(str, Enum):
    """Доступные инструменты агента."""
    SEARCH = "search"
    REFINE_AND_SEARCH = "refine_and_search"
    FINAL_ANSWER = "final_answer"
```

### ToolCall

```python
@dataclass(frozen=True)
class ToolCall:
    """Вызов инструмента, запрошенный агентом."""
    name: ToolName
    arguments: dict[str, Any]
```

### ToolResult

```python
@dataclass(frozen=True)
class ToolResult:
    """Результат выполнения инструмента."""
    tool_name: ToolName
    success: bool
    data: Any
    error: str | None = None
```

### SearchResult

```python
@dataclass(frozen=True)
class SearchResult:
    """Результат поиска из базы знаний."""
    content: str      # Текст сегмента
    source: str       # Путь к документу
    page: int | None  # Номер страницы (если есть)
    score: float      # Косинусное сходство (0-1)
```

### AgentState

```python
@dataclass
class AgentState:
    """Мутабельное состояние агента в процессе работы."""
    original_query: str                                    # Исходный вопрос
    search_history: list[tuple[str, list[SearchResult]]]  # История поисков
    reasoning_steps: list[str]                            # Шаги рассуждений
    iterations: int = 0                                   # Счётчик итераций
    final_answer: str | None = None                       # Финальный ответ

    def add_search(self, query: str, results: list[SearchResult]) -> None:
        """Добавить результаты поиска в историю."""

    def add_reasoning(self, step: str) -> None:
        """Добавить шаг рассуждения."""

    def get_all_results(self) -> list[SearchResult]:
        """Получить все уникальные результаты, отсортированные по score."""
```

### AgentResult

```python
@dataclass(frozen=True)
class AgentResult:
    """Финальный результат работы агента."""
    answer: str                    # Ответ
    sources: list[dict[str, Any]]  # Источники
    reasoning_steps: list[str]     # Шаги рассуждений
    search_count: int              # Количество поисков
    iterations: int                # Количество итераций
```

### Tool Protocol

```python
class Tool(Protocol):
    """Протокол для инструментов агента."""

    @property
    def name(self) -> ToolName: ...

    @property
    def description(self) -> str: ...

    async def execute(
        self,
        arguments: dict[str, Any],
        state: AgentState
    ) -> ToolResult: ...
```

---

## Код: tools.py

Реализация инструментов для агента.

### SearchTool

Семантический поиск по базе знаний.

```python
@dataclass
class SearchTool:
    db: Db                           # Подключение к БД
    embed_client: EmbeddingsClient   # Клиент для эмбеддингов
    embeddings_model: str            # Модель эмбеддингов
    top_k: int = 6                   # Количество результатов

    @property
    def name(self) -> ToolName:
        return ToolName.SEARCH

    async def execute(
        self,
        arguments: dict[str, Any],  # {"query": "..."}
        state: AgentState
    ) -> ToolResult:
        # 1. Получить query из arguments
        # 2. Создать эмбеддинг запроса
        # 3. Выполнить vector search в pgvector
        # 4. Обновить state.search_history
        # 5. Вернуть ToolResult с результатами
```

**Аргументы:**
| Имя | Тип | Описание |
|-----|-----|----------|
| `query` | string | Поисковый запрос |

**SQL запрос:**
```sql
SELECT s.content, d.source_path, d.title, s.page,
       (1 - (e.embedding <=> $query_vector)) as score
FROM segment_embeddings e
JOIN segments s ON s.id = e.segment_id
JOIN documents d ON d.id = s.document_id
ORDER BY (e.embedding <=> $query_vector) + 0
LIMIT $top_k
```

### RefineAndSearchTool

Поиск с переформулированным запросом.

```python
@dataclass
class RefineAndSearchTool:
    # Те же поля, что и SearchTool

    @property
    def name(self) -> ToolName:
        return ToolName.REFINE_AND_SEARCH

    async def execute(
        self,
        arguments: dict[str, Any],  # {"refined_query": "..."}
        state: AgentState
    ) -> ToolResult:
        # Аналогично SearchTool, но:
        # 1. Использует refined_query вместо query
        # 2. Добавляет reasoning step о переформулировке
```

**Аргументы:**
| Имя | Тип | Описание |
|-----|-----|----------|
| `refined_query` | string | Переформулированный запрос |

### FinalAnswerTool

Финализация ответа.

```python
@dataclass
class FinalAnswerTool:
    @property
    def name(self) -> ToolName:
        return ToolName.FINAL_ANSWER

    async def execute(
        self,
        arguments: dict[str, Any],  # {"answer": "..."}
        state: AgentState
    ) -> ToolResult:
        # 1. Извлечь answer из arguments
        # 2. Установить state.final_answer
        # 3. Вернуть успешный ToolResult
```

**Аргументы:**
| Имя | Тип | Описание |
|-----|-----|----------|
| `answer` | string | Финальный ответ пользователю |

### get_tools_schema()

Возвращает OpenAI-совместимую схему инструментов для LLM.

```python
def get_tools_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the knowledge base...",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "..."}
                    },
                    "required": ["query"]
                }
            }
        },
        # ... refine_and_search, final_answer
    ]
```

---

## Код: agent.py

Основной агент с ReAct loop.

### AgentConfig

```python
@dataclass
class AgentConfig:
    max_iterations: int = 3    # Макс. итераций
    top_k: int = 6             # Результатов на поиск
    include_sources: bool = True  # Включать источники
```

### Agent

```python
class Agent:
    def __init__(
        self,
        *,
        db: Db,
        embed_client: EmbeddingsClient,
        chat_client: ChatClient,
        settings: Settings,
        config: AgentConfig | None = None,
    ) -> None:
        # Инициализация инструментов
        self.search_tool = SearchTool(...)
        self.refine_tool = RefineAndSearchTool(...)
        self.final_answer_tool = FinalAnswerTool()

        self.tools_by_name = {
            "search": self.search_tool,
            "refine_and_search": self.refine_tool,
            "final_answer": self.final_answer_tool,
        }
```

### Agent.run()

Основной цикл агента.

```python
async def run(self, query: str) -> AgentResult:
    state = AgentState(original_query=query)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    while state.iterations < self.config.max_iterations:
        state.iterations += 1

        # 1. Вызов LLM с инструментами
        response = await self.chat_client.chat_completions({
            "model": self.settings.chat_model,
            "messages": messages,
            "tools": get_tools_schema(),
            "tool_choice": "auto",
        })

        # 2. Извлечение tool_calls из ответа
        message = response["choices"][0]["message"]
        tool_calls = message.get("tool_calls", [])

        # 3. Если нет tool_calls — завершение
        if not tool_calls:
            if message.get("content"):
                state.final_answer = message["content"]
            break

        # 4. Выполнение каждого tool_call
        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            tool_args = json.loads(tc["function"]["arguments"])

            tool = self.tools_by_name[tool_name]
            result = await tool.execute(tool_args, state)

            # Добавление результата в messages
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": self._format_tool_result(result),
            })

            # Если final_answer — выход
            if result.tool_name == ToolName.FINAL_ANSWER:
                break

        if state.final_answer:
            break

    return self._build_result(state)
```

### Системный промпт

```python
SYSTEM_PROMPT = """You are a research assistant with access to a knowledge base.
Your task is to answer the user's question using the available tools.

Strategy:
1. Start by searching the knowledge base with a query derived from the user's question.
2. Analyze the search results. If they don't fully answer the question,
   use refine_and_search with a different query angle.
3. You can perform up to {max_iterations} search iterations total.
4. Once you have enough information (or exhausted search options),
   use final_answer to respond.

Important:
- Base your answer ONLY on information from search results.
- If the knowledge base doesn't contain relevant information, say so honestly.
- Cite sources when providing information.
- Be concise but complete."""
```

---

## Диаграмма последовательности

```
User                API               Agent              LLM              Tools
  │                  │                  │                 │                 │
  │─── POST /agent/chat ───▶│          │                 │                 │
  │                  │──── run(query) ──▶│                │                 │
  │                  │                  │                 │                 │
  │                  │                  │── messages ────▶│                 │
  │                  │                  │◀── tool_calls ──│                 │
  │                  │                  │                 │                 │
  │                  │                  │─── search() ───────────────────▶│
  │                  │                  │◀── results ─────────────────────│
  │                  │                  │                 │                 │
  │                  │                  │── messages+results ─▶│           │
  │                  │                  │◀── refine_and_search ─│          │
  │                  │                  │                 │                 │
  │                  │                  │─── refine_and_search() ────────▶│
  │                  │                  │◀── results ─────────────────────│
  │                  │                  │                 │                 │
  │                  │                  │── messages+results ─▶│           │
  │                  │                  │◀── final_answer ────│            │
  │                  │                  │                 │                 │
  │                  │◀── AgentResult ──│                 │                 │
  │◀── Response ─────│                  │                 │                 │
```

---

## Расширение

### Добавление нового инструмента

1. Определить инструмент в `tools.py`:

```python
@dataclass
class MyNewTool:
    @property
    def name(self) -> ToolName:
        return ToolName.MY_NEW_TOOL  # Добавить в enum

    @property
    def description(self) -> str:
        return "Description for LLM"

    async def execute(
        self,
        arguments: dict[str, Any],
        state: AgentState
    ) -> ToolResult:
        # Реализация
        return ToolResult(...)
```

2. Добавить в `ToolName` enum в `protocol.py`

3. Добавить схему в `get_tools_schema()`

4. Зарегистрировать в `Agent.__init__()`:
```python
self.tools_by_name["my_new_tool"] = MyNewTool(...)
```

### Возможные расширения

- **Reranking** — добавить reranker после поиска для повышения precision
- **Hybrid search** — комбинировать vector + BM25 (keyword) search
- **Memory** — сохранять контекст между вызовами
- **Planning** — добавить инструмент для декомпозиции сложных вопросов
- **Streaming** — стриминг reasoning steps в реальном времени

---

## Конфигурация

Через environment variables (наследуются из основного API):

| Переменная | Описание | Default |
|------------|----------|---------|
| `TOP_K` | Количество результатов поиска | 6 |
| `CHAT_MODEL` | Модель LLM для агента | local-model |
| `EMBEDDINGS_MODEL` | Модель эмбеддингов | local-embedding-model |

Через request:

| Параметр | Описание | Default |
|----------|----------|---------|
| `max_iterations` | Макс. итераций агента | 3 |
| `citations` | Включить источники | false |

---

## Логирование

Агент логирует ключевые события:

```
INFO  agent_iteration=1 query=What is...
INFO  agent_tool_call tool=search args={'query': '...'}
INFO  agent_tool_call tool=refine_and_search args={'refined_query': '...'}
INFO  agent_complete iterations=2 searches=2 answer_len=450
```

Включить детальное логирование:
```bash
LOG_LEVEL=DEBUG
```

---

## Ограничения

1. **Латентность** — каждая итерация = вызов LLM, при max_iterations=3 это 3x время обычного RAG
2. **Стоимость** — больше токенов на запрос (tool calls + results в контексте)
3. **Tool calling** — требует модель с поддержкой function calling (GPT-4, Claude, Gemini, некоторые open-source)
4. **Нет streaming** — пока возвращает полный ответ, не стримит reasoning steps

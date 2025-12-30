# Phase 3: Query Analysis Agent - Implementation Guide

## Overview

This implementation provides a complete Query Analysis Agent for the Hadith RAG system. The agent serves as the entry point of the pipeline, transforming raw user queries into structured, retrieval-ready formats.

## Architecture

```
User Query
    ↓
┌─────────────────────────────────────────┐
│   Query Analysis Agent (FR-QAA-04)     │
│                                         │
│  1. Typo Correction Tool (FR-QAA-05)   │
│     ├─ Spelling correction              │
│     ├─ Arabic diacritics               │
│     └─ Language detection               │
│                                         │
│  2. Intent Classification (FR-QAA-07)  │
│     ├─ thematic_search                 │
│     ├─ specific_lookup                 │
│     └─ comparative_analysis            │
│                                         │
│  3. Query Decomposition (FR-QAA-08)    │
│     ├─ Complexity detection            │
│     └─ Sub-query generation            │
└─────────────────────────────────────────┘
    ↓
Structured Output → Retrieval Agent
```

## File Structure

```
src/
├── graph/
│   ├── __init__.py           # Graph module exports
│   └── state.py              # ✅ AgentState TypedDict definition
│
├── tools/
│   ├── __init__.py           # Tools module exports
│   └── query_processing.py  # ✅ All three tools with Pydantic models
│
├── agents/
│   └── query_analysis.py    # ✅ Main agent node + workflow builder
│
└── utils/
    └── llm_helper.py         # ✅ Modal LLM integration utilities

test_query_analysis.py        # ✅ Comprehensive test suite
```

## Implementation Details

### 1. State Management (`src/graph/state.py`)

```python
class AgentState(TypedDict):
    original_query: str
    corrected_query: Optional[str]
    query_intent: Optional[Literal["thematic_search", "specific_lookup", "comparative_analysis"]]
    sub_queries: Optional[List[str]]
    language: Optional[str]
    metadata: Optional[dict]
```

### 2. LLM Helper (`src/utils/llm_helper.py`)

**Features:**

- Async/sync HTTP calls to Modal Qwen2.5-14B endpoint
- Exponential backoff retry logic (3 attempts)
- JSON parsing with robust fallbacks
- LangSmith tracing integration
- Comprehensive error handling

**Key Functions:**

- `call_llm()` - Async LLM calls
- `call_llm_sync()` - Sync wrapper for non-async contexts
- `parse_json_response()` - Robust JSON extraction from LLM responses

### 3. Query Processing Tools (`src/tools/query_processing.py`)

#### Tool 1: Typo Correction (FR-QAA-05)

**Purpose:** Fix spelling errors and add Arabic diacritics

**Pydantic Model:**

```python
class TypoCorrectionOutput(BaseModel):
    corrected_text: str
    language: Literal["ar", "en", "mixed"]
    corrections_made: List[str]
```

**Features:**

- Handles Arabic and English
- Adds Arabic diacritics (تشكيل)
- Preserves original language
- Tracks all corrections made

#### Tool 2: Intent Classification (FR-QAA-07)

**Purpose:** Categorize query intent

**Pydantic Model:**

```python
class IntentClassificationOutput(BaseModel):
    intent: Literal["thematic_search", "specific_lookup", "comparative_analysis"]
    confidence: float  # 0.0 to 1.0
    reasoning: str
```

**Intent Categories:**

- `thematic_search`: Broad conceptual queries
- `specific_lookup`: Specific hadith by narrator/book/number
- `comparative_analysis`: Comparing topics or finding relationships

#### Tool 3: Query Decomposition (FR-QAA-08)

**Purpose:** Break complex queries into atomic sub-queries

**Pydantic Model:**

```python
class QueryDecompositionOutput(BaseModel):
    is_complex: bool
    sub_queries: List[str]
    reasoning: str
```

**Features:**

- Detects multi-part questions
- Generates self-contained sub-queries
- Preserves original language per part

### 4. Query Analysis Agent (`src/agents/query_analysis.py`)

**Main Function:** `query_analysis_agent(state: AgentState) -> Dict[str, Any]`

**Workflow:**

1. Validates input
2. Runs typo correction → updates state
3. Runs intent classification → updates state
4. Runs query decomposition → updates state
5. Returns complete state update with metadata

**Error Handling:**

- Graceful fallbacks at each stage
- All errors logged in metadata
- Pipeline continues even if stages fail

**Observability:**

- All operations traced via LangSmith
- Detailed metadata tracking
- Comprehensive logging

## Usage

### Direct Function Call

```python
from src.agents.query_analysis import analyze_query

result = analyze_query("What are hadiths about prayer?")

print(result['corrected_query'])  # Corrected text
print(result['query_intent'])     # Intent classification
print(result['sub_queries'])      # Sub-queries (if complex)
print(result['language'])         # Detected language
```

### LangGraph Workflow

```python
from src.agents.query_analysis import create_query_analysis_workflow

# Create the graph
graph = create_query_analysis_workflow()

# Invoke with initial state
result = graph.invoke({
    "original_query": "What are hadiths about prayer?",
    "corrected_query": None,
    "query_intent": None,
    "sub_queries": None,
    "language": None,
    "metadata": {}
})
```

### Integration with Full RAG Pipeline

```python
from langgraph.graph import StateGraph, START, END
from src.graph.state import AgentState
from src.agents.query_analysis import query_analysis_agent

builder = StateGraph(AgentState)

# Add nodes
builder.add_node("query_analysis", query_analysis_agent)
builder.add_node("retrieval", retrieval_agent)        # Your retrieval agent
builder.add_node("synthesis", synthesis_agent)        # Your synthesis agent

# Connect edges
builder.add_edge(START, "query_analysis")
builder.add_edge("query_analysis", "retrieval")
builder.add_edge("retrieval", "synthesis")
builder.add_edge("synthesis", END)

graph = builder.compile()
```

## Testing

### Run Test Suite

```bash
# Activate virtual environment
.\venv\Scripts\activate

# Run tests
python test_query_analysis.py
```

### Test Cases Included

1. ✅ Simple English query
2. ✅ English query with typos
3. ✅ Specific lookup query
4. ✅ Complex multi-part query
5. ✅ Simple Arabic query
6. ✅ Complex Arabic query
7. ✅ Specific Arabic hadith lookup
8. ✅ Mixed language query

### Expected Output

```
🚀 🚀 🚀 ... (decorative header)
Query Analysis Agent - Phase 3 Test Suite
🚀 🚀 🚀 ...

================================================================================
  Test 1/8: Simple English Query
================================================================================

Query: "What are hadiths about prayer?"

📝 Original Query:  What are hadiths about prayer?
✅ Corrected Query: What are hadiths about prayer?
🌐 Language:        en
🎯 Intent:          thematic_search
🔍 Complex Query:   No
📊 Confidence:      95.0%

📋 Validation:
  ✓ Intent matches expected: thematic_search
  ✓ Complexity matches expected: False

... (more tests)

================================================================================
  Test Summary
================================================================================

✅ Passed: 8/8
❌ Failed: 0/8
📊 Success Rate: 100.0%

🎉 All tests passed!
```

## Configuration

### LLM Endpoint

Configured in `src/utils/llm_helper.py`:

```python
QWEN_ENDPOINT = "https://sazaitet110--qwen2-5-14b-runner-qwenmodel-generate.modal.run"
```

### Request Parameters

```python
DEFAULT_TIMEOUT = 60.0      # seconds
MAX_RETRIES = 3             # retry attempts
RETRY_BACKOFF = 2.0         # exponential backoff multiplier
```

### Tool Temperatures

- Typo Correction: `0.3` (consistent corrections)
- Intent Classification: `0.2` (stable classification)
- Query Decomposition: `0.3` (balanced creativity)

## Error Handling

All tools implement graceful fallbacks:

### Typo Correction Fallback

```python
# If fails: return original query with no corrections
corrected_query = original_query
language = "en"
```

### Intent Classification Fallback

```python
# If fails: default to thematic_search (most common)
query_intent = "thematic_search"
confidence = 0.5
```

### Query Decomposition Fallback

```python
# If fails: treat as simple query
sub_queries = None
is_complex = False
```

## Observability

### LangSmith Tracing

All functions are decorated with `@traceable`:

```python
@traceable(name="query_analysis_agent")
def query_analysis_agent(state: AgentState):
    # ...
```

This enables:

- Individual tool execution visibility
- Input/output tracking
- Performance monitoring
- Error debugging

### Metadata Tracking

All operations store metadata:

```python
metadata = {
    "query_analysis": {
        "stages_completed": ["typo_correction", "intent_classification", ...],
        "typo_correction": {
            "original": "...",
            "corrected": "...",
            "corrections_made": [...]
        },
        "intent_classification": {
            "intent": "thematic_search",
            "confidence": 0.95,
            "reasoning": "..."
        },
        "errors": [
            {"stage": "...", "error": "..."}
        ]
    }
}
```

## Next Steps

### Phase 4: Retrieval Agent

Connect this agent to the retrieval system:

```python
def retrieval_agent(state: AgentState):
    # Use state['corrected_query']
    # Use state['query_intent'] to select retrieval strategy
    # Use state['sub_queries'] for multi-step retrieval
    pass
```

### LangSmith Setup (Optional)

Add to `.env`:

```bash
LANGSMITH_API_KEY=your_key_here
LANGSMITH_PROJECT=hadith-rag
```

## Troubleshooting

### Import Errors

Ensure all `__init__.py` files exist:

```bash
src/__init__.py
src/graph/__init__.py
src/tools/__init__.py
src/agents/__init__.py
src/utils/__init__.py
```

### Modal Endpoint Not Responding

Check endpoint availability:

```python
import httpx

response = httpx.get("https://sazaitet110--qwen2-5-14b-runner-qwenmodel-generate.modal.run")
print(response.status_code)
```

### JSON Parsing Failures

The `parse_json_response()` function handles:

- Direct JSON parsing
- Markdown code block extraction
- Fallback value support

If issues persist, check LLM prompts for clarity.

## Definition of Done ✅

- [x] `src/graph/state.py` - AgentState TypedDict
- [x] `src/utils/llm_helper.py` - Modal LLM integration
- [x] `src/tools/query_processing.py` - All 3 tools with Pydantic
- [x] `src/agents/query_analysis.py` - Main agent node
- [x] Error handling with graceful fallbacks
- [x] LangSmith tracing integration
- [x] Comprehensive test suite
- [x] Documentation

## Performance Considerations

- **Latency**: ~2-5 seconds per query (3 LLM calls)
- **Token Usage**: ~500-1500 tokens per query
- **Parallelization**: Tools run sequentially (required for dependencies)
- **Caching**: Consider caching common corrections/intents

## License

Part of the Hadith RAG System project.

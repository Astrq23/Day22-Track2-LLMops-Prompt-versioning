# Lab Requirements — Day 22: LangSmith + Prompt Versioning

## Python Version
Python 3.10 or higher

## Install All Dependencies

```bash
pip install -r requirements.txt
```

## requirements.txt

```
langchain>=0.3.0
langchain-core>=0.3.0
langchain-ollama>=0.2.0
langchain-community>=0.3.0
langchain-text-splitters>=0.3.0
langsmith>=0.2.0
ollama>=0.4.0
faiss-cpu>=1.7.0
ragas>=0.4.0
guardrails-ai>=0.5.0
python-dotenv>=1.0.0
tiktoken>=0.5.0
datasets>=2.0.0
numpy>=1.25.0
```

## Package Purposes

| Package | Used For |
|---------|---------|
| `langchain` | Core LLM framework |
| `langchain-ollama` | ChatOllama, OllamaEmbeddings |
| `langchain-community` | FAISS vectorstore integration |
| `langchain-text-splitters` | RecursiveCharacterTextSplitter |
| `langsmith` | LangSmith tracing, Prompt Hub client |
| `ollama` | Ollama local model runtime client |
| `faiss-cpu` | Similarity search index |
| `ragas` | RAG evaluation metrics |
| `guardrails-ai` | Output validation framework |
| `python-dotenv` | Load `.env` file |
| `tiktoken` | Token counting for text splitters |
| `datasets` | Required by RAGAS internally |
| `numpy` | Averaging RAGAS score lists |

## Important Version Notes

### RAGAS 0.4.x
- Use `from ragas.metrics import faithfulness, answer_relevancy, ...` (NOT from `ragas.metrics.collections`)
- `result[metric_name]` returns a **list** of floats for multiple samples — use `numpy.mean()` to average
- Pass `llm=` and `embeddings=` to the `evaluate()` function, not to metric constructors

### Guardrails AI 0.10.x
- `on_fail` parameter belongs in the **validator constructor**: `MyValidator(on_fail=OnFailAction.FIX)`
- `Guard.use()` accepts validator **instances**, not classes
- `Guard.validate(text)` is the main entry point

### LangChain 0.3.x
- Use `ChatOllama(model=..., base_url=...)` for local Ollama chat models
- Use `OllamaEmbeddings(model=..., base_url=...)` for local Ollama embeddings

## Environment Variables

Copy this to your `.env` file:


> ⚠️ **Never commit `.env` to git.** Add it to `.gitignore`.

## Verify Installation

Run the config check:
```bash
python config.py
```

Expected output:
```
✅ Config loaded successfully
   LangSmith project : your-project-name
   Ollama endpoint   : http://localhost:11434
   Default LLM model : llama3.1:8b
   Embedding model   : nomic-embed-text
```

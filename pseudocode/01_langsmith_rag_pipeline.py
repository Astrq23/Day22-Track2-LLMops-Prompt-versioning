"""Step 1 - LangSmith-instrumented RAG pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable

from lab_common import PROMPT_V1, build_vectorstore, ensure_environment, format_docs, make_llm
from qa_pairs import SAMPLE_QUESTIONS


ensure_environment()
llm = make_llm()


def build_rag_chain(vectorstore):
    """Build a LangChain RAG chain using LCEL."""

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT_V1
        | llm
        | StrOutputParser()
    )
    return chain, retriever


@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    """Invoke the chain and return the answer."""

    return chain.invoke(question)


def main():
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)

    vectorstore = build_vectorstore()
    chain, _ = build_rag_chain(vectorstore)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        answer = ask(chain, question)
        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question[:60]}")
        print(f"       A: {answer[:100]}\n")

    print(f"✅ {len(SAMPLE_QUESTIONS)} traces sent to LangSmith project '{os.environ.get('LANGCHAIN_PROJECT', 'day22-langsmith-lab')}'")
    print("   Open https://smith.langchain.com to view traces.")


if __name__ == "__main__":
    main()

"""Step 2 - Prompt Hub and A/B routing."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable

from lab_common import PROMPT_V1, PROMPT_V2, build_vectorstore, ensure_environment, make_llm
from qa_pairs import SAMPLE_QUESTIONS


ensure_environment()

PROMPT_V1_NAME = "day22-rag-prompt-v1"
PROMPT_V2_NAME = "day22-rag-prompt-v2"


def push_prompts_to_hub(client):
    """Upload both prompt versions to LangSmith Prompt Hub."""

    try:
        url = client.push_prompt(PROMPT_V1_NAME, object=PROMPT_V1, description="V1 - concise answers")
        print(f"✅ Pushed V1 -> {url}")
    except Exception as exc:
        print(f"⚠️  V1 push skipped: {exc}")

    try:
        url = client.push_prompt(PROMPT_V2_NAME, object=PROMPT_V2, description="V2 - structured answers")
        print(f"✅ Pushed V2 -> {url}")
    except Exception as exc:
        print(f"⚠️  V2 push skipped: {exc}")


def pull_prompts_from_hub(client):
    """Download both prompt versions from LangSmith Prompt Hub."""

    prompts = {}
    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"↓ Pulled '{PROMPT_V1_NAME}' from Hub")
    except Exception:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"ℹ️  Using local fallback for '{PROMPT_V1_NAME}'")

    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"↓ Pulled '{PROMPT_V2_NAME}' from Hub")
    except Exception:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"ℹ️  Using local fallback for '{PROMPT_V2_NAME}'")

    return prompts


def get_prompt_version(request_id: str) -> str:
    """Deterministically route a request to V1 or V2 using MD5."""

    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """Run the RAG chain using the given prompt version."""

    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
    return {"question": question, "answer": answer, "version": version}


def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)

    client = Client(api_key=os.environ.get("LANGCHAIN_API_KEY"))
    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client)

    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = make_llm()

    v1_count = 0
    v2_count = 0

    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt = prompts[version_key]

        ask_ab(retriever, llm, prompt, question, version_tag)
        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1
        print(f"[{i + 1:02d}] [prompt-{version_tag}] {question[:55]}...")

    print(f"\nRouting summary: V1={v1_count}, V2={v2_count}")


if __name__ == "__main__":
    main()

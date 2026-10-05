"""Step 5: Give the retrieved chunks + question to a language model.

Backend is picked automatically:
  1. ANTHROPIC_API_KEY set -> Claude
  2. OPENAI_API_KEY set    -> OpenAI
  3. otherwise             -> free local model (flan-t5), no key needed
"""
import os

NOT_FOUND = "I could not find this in the document."


def build_prompt(question: str, contexts: list[dict]) -> str:
    context_text = "\n\n".join(f"[Page {c['page']}] {c['text']}" for c in contexts)
    return (
        "Answer the question using ONLY the context below. "
        f"If the answer is not in the context, reply exactly: \"{NOT_FOUND}\"\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\nAnswer:"
    )


def _claude(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    model = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
    msg = client.messages.create(
        model=model, max_tokens=700, messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text.strip()


def _openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=0
    )
    return resp.choices[0].message.content.strip()


_local_pipe = None


def _local(prompt: str) -> str:
    global _local_pipe
    if _local_pipe is None:
        from transformers import pipeline
        _local_pipe = pipeline("text2text-generation", model="google/flan-t5-base")
    out = _local_pipe(prompt, max_new_tokens=200, truncation=True)
    return out[0]["generated_text"].strip()


def generate_answer(question: str, contexts: list[dict]) -> str:
    if not contexts:
        return NOT_FOUND
    prompt = build_prompt(question, contexts)
    if os.getenv("ANTHROPIC_API_KEY"):
        return _claude(prompt)
    if os.getenv("OPENAI_API_KEY"):
        return _openai(prompt)
    return _local(prompt)

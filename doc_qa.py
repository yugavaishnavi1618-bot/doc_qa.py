import argparse
import getpass
import math
import os
import re
import sys
from collections import Counter

import anthropic

DEFAULT_MODEL = "claude-haiku-5-5"  # small, fast, low-cost model


# ----------------------------- 1. Load document -----------------------------
def load_document(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()

    if ext in (".txt", ".md"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)

    if ext == ".docx":
        from docx import Document
        doc = Document(path)
        return "\n\n".join(p.text for p in doc.paragraphs)

    raise ValueError(f"Unsupported file type: {ext}. Use .txt, .md, .pdf or .docx")


# --------------------------- 2. Split into paragraphs -----------------------
def split_into_paragraphs(text: str, min_chars: int = 40, max_chars: int = 1200):
    raw = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    # PDFs often have no blank lines, so group sentences instead
    if len(raw) <= 1:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        raw, buf = [], ""
        for s in sentences:
            if len(buf) + len(s) > max_chars // 2 and buf:
                raw.append(buf.strip())
                buf = ""
            buf += " " + s
        if buf.strip():
            raw.append(buf.strip())

    paragraphs, buffer = [], ""
    for p in raw:
        p = re.sub(r"\s+", " ", p)
        if len(p) < min_chars:          # merge tiny lines like titles
            buffer += " " + p
            continue
        p = (buffer + " " + p).strip()
        buffer = ""
        while len(p) > max_chars:       # break overly long paragraphs
            cut = p.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut != -1 else max_chars
            paragraphs.append(p[:cut].strip())
            p = p[cut:].strip()
        paragraphs.append(p)
    if buffer.strip():
        paragraphs.append(buffer.strip())
    return paragraphs


# ------------------------ 3. Retriever (pure-Python BM25) -------------------
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "what", "which", "who",
    "whom", "how", "when", "where", "why", "do", "does", "did", "can", "could",
    "should", "would", "will", "about", "into", "than", "then", "there", "their",
    "i", "you", "he", "she", "we", "they", "me", "my", "your", "our", "his", "her",
}


def tokenize(text: str):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


class Retriever:
    def __init__(self, paragraphs, k1: float = 1.5, b: float = 0.75):
        self.paragraphs = paragraphs
        self.k1, self.b = k1, b

        docs = [tokenize(p) for p in paragraphs]
        self.tf = [Counter(d) for d in docs]
        self.doc_len = [len(d) for d in docs]
        self.avg_len = max(sum(self.doc_len) / len(docs), 1)

        df = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, top_k: int = 2):
        q_terms = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tf):
            score = 0.0
            for t in q_terms:
                f = tf.get(t, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                score += self.idf[t] * f * (self.k1 + 1) / denom
            scores.append(score)

        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(i, self.paragraphs[i], scores[i]) for i in top_idx]


# --------------------------- 4. Generate the answer -------------------------
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly using the "
    "provided document excerpts. If the answer is not contained in the excerpts, "
    "say: \"I couldn't find the answer in the document.\" Be concise and accurate."
)


def generate_answer(client, model, question, retrieved):
    context = "\n\n".join(
        f"[Paragraph {idx + 1}]\n{text}" for idx, text, _ in retrieved
    )
    user_prompt = (
        f"Document excerpts:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer based only on the excerpts above."
    )
    response = client.messages.create(
        model=model,
        max_tokens=500,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


# ------------------------------- 5. API key ---------------------------------
def get_api_key() -> str:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        key = getpass.getpass("Enter your Anthropic API key (input hidden): ").strip()
    if not key:
        sys.exit("No API key provided.")
    return key


# ------------------------------- 6. Main loop -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Ask questions about a document.")
    parser.add_argument("document", help="Path to .txt, .md, .pdf or .docx file")
    parser.add_argument("--top_k", type=int, default=2, help="Paragraphs to retrieve")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model name")
    args = parser.parse_args()

    client = anthropic.Anthropic(api_key=get_api_key())

    print(f"\nLoading '{args.document}' ...")
    text = load_document(args.document)
    paragraphs = split_into_paragraphs(text)
    if not paragraphs:
        sys.exit("Could not extract any text from the document.")
    retriever = Retriever(paragraphs)
    print(f"Indexed {len(paragraphs)} paragraphs. Type 'exit' to quit.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        retrieved = retriever.search(question, top_k=args.top_k)

        print("\n" + "=" * 70)
        print("RETRIEVED PARAGRAPH(S)")
        print("=" * 70)
        for idx, para, score in retrieved:
            print(f"\n[Paragraph {idx + 1} | relevance score: {score:.2f}]\n{para}")

        try:
            answer = generate_answer(client, args.model, question, retrieved)
        except anthropic.AuthenticationError:
            sys.exit("\nInvalid API key. Please check it and try again.")
        except anthropic.APIError as e:
            print(f"\nAPI error: {e}\n")
            continue

        print("\n" + "=" * 70)
        print("GENERATED ANSWER")
        print("=" * 70)
        print(f"\n{answer}\n")


if __name__ == "__main__":
    main()

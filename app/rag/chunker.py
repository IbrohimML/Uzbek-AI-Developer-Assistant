import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = BASE_DIR / "data" / "processed" / "documents.json"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "chunks.json"


def load_documents():
    with open(INPUT_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    all_chunks = []

    for document_index, document in enumerate(documents):
        text = document["content"]

        chunks = splitter.create_documents(
            texts=[text],
            metadatas=[{
                "title": document["title"],
                "category": document["category"],
                "source": document["source"],
            }],
        )

        for index, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"document_{document_index}_chunk_{index}",
                "title": chunk.metadata["title"],
                "category": chunk.metadata["category"],
                "source": chunk.metadata["source"],
                "start_index": chunk.metadata["start_index"],
                "content": chunk.page_content,
            })

    return all_chunks


def save_chunks(chunks):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=4)

    print(f"\nSuccessfully created {len(chunks)} chunks!")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    documents = load_documents()

    print(f"Loaded documents: {len(documents)}")

    chunks = create_chunks(documents)

    save_chunks(chunks)

    if chunks:
        print("\nFirst chunk preview:")
        print(chunks[0]["content"][:500])
import json
import numpy as np
import chromadb

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"
EMBEDDINGS_PATH = BASE_DIR / "data" / "processed" / "embeddings.npy"
DB_PATH = BASE_DIR / "data" / "chroma_db"

COLLECTION_NAME = "documents"


def load_data():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    embeddings = np.load(EMBEDDINGS_PATH)

    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings count do not match!")

    return chunks, embeddings


def create_vector_store(chunks, embeddings):
    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    # Eski collection bo'lsa, o'chiramiz
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Old collection deleted.")
    except Exception:
        pass

    # Yangi collection yaratamiz
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],

        embeddings=embeddings.tolist(),

        documents=[
            chunk["content"] for chunk in chunks
        ],

        metadatas=[
            {
                "title": chunk["title"],
                "category": chunk["category"],
                "source": chunk["source"],
                "start_index": chunk["start_index"],
            }
            for chunk in chunks
        ]
    )

    return collection


if __name__ == "__main__":
    chunks, embeddings = load_data()

    print(f"Loaded chunks: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")

    collection = create_vector_store(chunks, embeddings)

    print("\nVector database created successfully!")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Total vectors: {collection.count()}")
    print(f"Database path: {DB_PATH}")
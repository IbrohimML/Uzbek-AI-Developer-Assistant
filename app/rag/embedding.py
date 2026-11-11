import json
import numpy as np
import torch

from pathlib import Path
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = BASE_DIR / "data" / "processed" / "chunks.json"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "embeddings.npy"

MODEL_NAME = "BAAI/bge-m3"


def load_chunks():
    with open(INPUT_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def create_embeddings(chunks):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {device}")

    model = SentenceTransformer(
        MODEL_NAME,
        device=device
    )

    texts = [chunk["content"] for chunk in chunks]

    print(f"\nCreating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        batch_size=8,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embeddings


def save_embeddings(embeddings):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_PATH, embeddings)

    print(f"\nEmbeddings saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    chunks = load_chunks()

    print(f"Loaded chunks: {len(chunks)}")

    embeddings = create_embeddings(chunks)

    save_embeddings(embeddings)

    print("\nEmbedding information:")
    print("Shape:", embeddings.shape)
    print("Data type:", embeddings.dtype)
    print("Total vectors:", len(embeddings))
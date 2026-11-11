import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "dataset.json"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "documents.json"


def load_documents():
    """data/raw ichidagi JSON datasetni o'qiydi."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset topilmadi: {RAW_DATA_PATH}"
        )

    with open(RAW_DATA_PATH, "r", encoding="utf-8") as file:
        documents = json.load(file)

    if not isinstance(documents, list):
        raise ValueError("Dataset JSON list formatida bo'lishi kerak.")

    for document in documents:
        if not document.get("content", "").strip():
            raise ValueError("Har bir document content maydoniga ega bo'lishi kerak.")

    print(f"Successfully loaded: {len(documents)} documents")

    return documents


def save_documents(documents):
    """O'qilgan datasetni processed papkasiga saqlaydi."""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=4
        )

    print(f"Saved {len(documents)} documents")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    documents = load_documents()
    save_documents(documents)
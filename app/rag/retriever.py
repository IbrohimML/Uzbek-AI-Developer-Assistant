import chromadb
import torch

from pathlib import Path
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[2]

DB_PATH = BASE_DIR / "data" / "chroma_db"
COLLECTION_NAME = "documents"
MODEL_NAME = "BAAI/bge-m3"

TOP_K = 5
MAX_DISTANCE = 0.55


# Oddiy so'zlarni standart shaklga o'tkazish
SYNONYMS = {
    "qande": "qanday",
    "qanaqa": "qanday",
    "nme": "nima",
    "nech pul": "narxi qancha",
    "nechpul": "narxi qancha",
    "ayfon": "iphone",
    "dostavka": "yetkazib berish",
    "dostafka": "yetkazib berish",
    "skidka": "chegirma",
    "vozvrat": "qaytarish",
    "obmen": "almashtirish",
    "products": "mahsulotlar",
    "product": "mahsulot",
    "sz": "siz",
}


def normalize_query(query: str) -> str:
    query = query.lower().strip()

    for old, new in SYNONYMS.items():
        query = query.replace(old, new)

    return query


class Retriever:

    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading embedding model on {self.device}...")

        self.model = SentenceTransformer(
            MODEL_NAME,
            device=self.device
        )

        self.client = chromadb.PersistentClient(
            path=str(DB_PATH)
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

        print(
            f"Database loaded: {self.collection.count()} chunks"
        )

    # Oddiy semantic search
    def search(self, query: str, top_k: int = TOP_K):

        total_chunks = self.collection.count()

        if total_chunks == 0:
            return []

        normalized_query = normalize_query(query)

        print(f"Original query: {query}")
        print(f"Normalized query: {normalized_query}")

        query_embedding = self.model.encode(
            normalized_query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, total_chunks),
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        retrieved_documents = []

        for i in range(len(results["documents"][0])):

            distance = results["distances"][0][i]

            if distance > MAX_DISTANCE:
                continue

            retrieved_documents.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": distance
            })

        return retrieved_documents

    # Kategoriya bo'yicha barcha ma'lumotlarni olish
    def search_by_category(self, category="Products"):

        results = self.collection.get(
            where={"category": category},
            include=[
                "documents",
                "metadatas"
            ]
        )

        documents = []

        for content, metadata in zip(
            results["documents"],
            results["metadatas"]
        ):

            documents.append({
                "content": content,
                "metadata": metadata,
                "distance": 0
            })

        return documents
    def search_by_source(self, source):
        results = self.collection.get(
        where={"source": source},
        include=["documents", "metadatas"]
        )

        documents = []

        for content, metadata in zip(
            results["documents"],
            results["metadatas"]
        ):
            documents.append({
                "content": content,
                "metadata": metadata,
                "distance": 0
            })

        return documents


if __name__ == "__main__":

    retriever = Retriever()

    while True:

        query = input(
            "\nSavolingiz (exit — chiqish): "
        )

        if query.lower() == "exit":
            break

        if not query.strip():
            continue

        documents = retriever.search(query)

        print("\n" + "=" * 60)

        if not documents:
            print("Bazadan mos ma'lumot topilmadi.")
            print("=" * 60)
            continue

        print(f"TOP {len(documents)} RESULTS")
        print("=" * 60)

        for index, document in enumerate(
            documents,
            start=1
        ):

            print(f"\nResult #{index}")
            print("Title:", document["metadata"]["title"])
            print("Category:", document["metadata"]["category"])
            print("Distance:", round(document["distance"], 4))
            print("Source:", document["metadata"]["source"])

            print("\nContent:")
            print(document["content"][:700])

            print("-" * 60)
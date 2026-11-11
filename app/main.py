from app.rag.retriever import Retriever
from app.rag.generator import (
    generate_response,
    rewrite_query,
    NO_ANSWER
)


ALL_PRODUCTS_KEYWORDS = [
    "arzonroq",
    "arzonrogi",
    "arzonrog'i",
    "eng arzon",
    "mahsulotlaringiz",
    "qanday mahsulot",
    "nimalar bor",
    "barcha mahsulot",
    "boshqa variant",
    "alternativa",
    "variantlari"
]


DISCOUNT_KEYWORDS = [
    "skidka",
    "chegirma",
    "arzonroq",
    "arzonlashtir",
    "aksiya"
]


def needs_all_products(question):
    question = question.lower()

    return any(
        keyword in question
        for keyword in ALL_PRODUCTS_KEYWORDS
    )


def needs_discount_policy(question):
    question = question.lower()

    return any(
        keyword in question
        for keyword in DISCOUNT_KEYWORDS
    )


def is_greeting(question):
    greetings = [
        "assalomu aleykum",
        "assalomu alaykum",
        "salom",
        "hello",
        "hi"
    ]

    return question.lower().strip() in greetings


def main():
    print("=" * 60)
    print("TECHSTORE AI ASSISTANT")
    print("=" * 60)

    print("\nInitializing RAG system...")
    retriever = Retriever()

    chat_history = []

    print("\nSystem is ready!")
    print("Savol bering. Chiqish uchun 'exit' yozing.")

    while True:
        question = input("\nYou: ").strip()

        if not question:
            continue

        if question.lower() == "exit":
            print("\nAssistant: Goodbye!")
            break

        if is_greeting(question):
            answer = (
                "Va alaykum assalom! "
                "TechStore AI yordamchisiga xush kelibsiz. "
                "Sizga qanday yordam bera olaman?"
            )

            print("\nAssistant:", answer)
            continue

        try:
            if needs_discount_policy(question):
                print("\nSearching discount policy...")

                documents = retriever.search_by_source(
                    source="discount_policy"
                )

            elif needs_all_products(question):
                print("\nSearching all products...")

                documents = retriever.search_by_category(
                    category="Products"
                )

            else:
                print("\nUnderstanding your question...")

                search_query = rewrite_query(
                    question=question,
                    chat_history=chat_history
                )

                print(f"Search query: {search_query}")
                print("\nSearching knowledge base...")

                documents = retriever.search(
                    query=search_query,
                    top_k=5
                )

            print("\nGenerating answer with Gemma 4...")

            answer = generate_response(
                question=question,
                retrieved_documents=documents,
                chat_history=chat_history
            )

            print("\n" + "=" * 60)
            print("ASSISTANT")
            print("=" * 60)
            print(answer)

            chat_history.append({
                "role": "user",
                "content": question
            })

            chat_history.append({
                "role": "assistant",
                "content": answer
            })

            chat_history = chat_history[-10:]

            if documents and answer != NO_ANSWER:
                print("\nSOURCES")
                print("-" * 60)

                seen_sources = set()

                for document in documents:
                    metadata = document["metadata"]
                    source = metadata["source"]

                    if source not in seen_sources:
                        seen_sources.add(source)

                        print(f"Title: {metadata['title']}")
                        print(f"Source: {source}")
                        print()

        except Exception as error:
            print(f"\nError: {error}")


if __name__ == "__main__":
    main()

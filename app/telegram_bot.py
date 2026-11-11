import os
import asyncio
import logging

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

from app.rag.retriever import Retriever
from app.rag.generator import (
    generate_response,
    rewrite_query,
    NO_ANSWER
)


# Environment variables
load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


# Logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


# Keywords
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


# Initialize RAG
print("Initializing RAG system...")

retriever = Retriever()

print("RAG system is ready!")


# Chat history for each Telegram user
chat_histories = {}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    welcome_message = (
        "Assalomu alaykum! 👋\n\n"
        "TechStore AI yordamchisiga xush kelibsiz!\n\n"
        "Men sizga quyidagilar bo'yicha yordam bera olaman:\n"
        "📱 Mahsulotlar va narxlar\n"
        "🚚 Yetkazib berish\n"
        "💳 To'lov usullari\n"
        "🛡 Kafolat va qaytarish\n"
        "🎁 Chegirmalar\n\n"
        "Savolingizni yozishingiz mumkin!"
    )

    await update.message.reply_text(welcome_message)


def process_question(question, chat_history):

    if needs_discount_policy(question):

        documents = retriever.search_by_source(
            source="discount_policy"
        )

    elif needs_all_products(question):

        documents = retriever.search_by_category(
            category="Products"
        )

    else:

        search_query = rewrite_query(
            question=question,
            chat_history=chat_history
        )

        documents = retriever.search(
            query=search_query,
            top_k=5
        )

    answer = generate_response(
        question=question,
        retrieved_documents=documents,
        chat_history=chat_history
    )

    return answer


async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    user_id = update.effective_user.id
    question = update.message.text.strip()

    if not question:
        return

    if is_greeting(question):

        await update.message.reply_text(
            "Va alaykum assalom! 👋\n"
            "TechStore AI yordamchisiga xush kelibsiz. "
            "Sizga qanday yordam bera olaman?"
        )

        return

    # Create chat history if it does not exist
    if user_id not in chat_histories:
        chat_histories[user_id] = []

    chat_history = chat_histories[user_id]

    # Show typing status
    await update.message.chat.send_action("typing")

    try:

        answer = await asyncio.to_thread(
            process_question,
            question,
            chat_history
        )

        await update.message.reply_text(answer[:4000])

        # Save conversation history
        chat_history.append({
            "role": "user",
            "content": question
        })

        chat_history.append({
            "role": "assistant",
            "content": answer
        })

        chat_histories[user_id] = chat_history[-10:]

    except Exception as error:

        logging.exception("Message processing failed: %s", error)

        await update.message.reply_text(
            "Kechirasiz, hozir texnik xatolik yuz berdi. "
            "Iltimos, birozdan keyin qayta urinib ko'ring."
        )


def main():

    if not TOKEN:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN .env faylida topilmadi!"
        )

    application = (
        Application.builder()
        .token(TOKEN)
        .build()
    )

    application.add_handler(
        CommandHandler("start", start)
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message
        )
    )

    print("Telegram bot is running...")

    application.run_polling()


if __name__ == "__main__":
    main()
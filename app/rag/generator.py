import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "gemma4:e4b"

NO_ANSWER = (
    "Kechirasiz, bu savolga javob berish uchun "
    "bazamda yetarli ma'lumot yo'q.\n\n"
    "Batafsil ma'lumot olish uchun menejerimizga bog'laning:\n"
    "Telefon: +998 00 000 00 00\n"
    "Telegram: @TECHSTORE_MANAGER_USERNAME"
)


SYSTEM_PROMPT = """
Sen TechStore online do'konining AI yordamchisisan.

QOIDALAR:
1. Faqat berilgan CONTEXT ma'lumotlaridan foydalan.
2. Mahsulot narxi va xususiyatlarini to'qima.
3. Oldingi suhbatdan mijoz nimani nazarda tutayotganini tushun.
4. Javoblarni o'zbek tilida, samimiy va qisqa yoz.
5. Agar ma'lumot yetarli bo'lmasa, buni ochiq ayt.
6. Javobda manbani ko'rsat.
"""


def ask_gemma(messages):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0
            }
        },
        timeout=180
    )

    response.raise_for_status()

    return response.json()["message"]["content"].strip()


def rewrite_query(question, chat_history):
    if not chat_history:
        return question

    messages = [
        {
            "role": "system",
            "content": """
Sen TechStore do'koni uchun qidiruv so'rovlarini
yaxshilovchi yordamchisan.

VAZIFANG:
Mijoz savolini qidiruv uchun aniq va tushunarli shaklga keltirish.

MUHIM QOIDALAR:
1. Savolning asl ma'nosini hech qachon o'zgartirma.
2. Mijoz "dastavka", "dostavka" desa, bu YETKAZIB BERISH degani.
3. Mijoz "kafolat" desa, bu WARRANTY degani.
4. Mijoz "skidka", "arzonroq" desa, bu CHEGIRMA degani.
5. Mijoz "qancha", "nech pul" desa, bu NARX degani.
6. Oldingi suhbatdagi mahsulot nomini faqat yangi savol
   unga bog'liq bo'lsa ishlat.
7. Yangi mavzudagi savolni eski suhbatga bog'lama.
8. Hech qanday yangi ma'lumot to'qima.
9. Faqat bitta qidiruv savolini qaytar.
10. Izoh yozma.
"""
        },
        {
            "role": "user",
            "content": f"""
OLDINGI SUHBAT:
{chat_history}

YANGI SAVOL:
{question}

Qidiruv uchun savol:
"""
        }
    ]

    return ask_gemma(messages)


def generate_response(
    question: str,
    retrieved_documents: list[dict],
    chat_history=None
):

    if not retrieved_documents:
        return NO_ANSWER

    context_parts = []

    for index, document in enumerate(
        retrieved_documents,
        start=1
    ):

        metadata = document["metadata"]

        context_parts.append(
            f"""
SOURCE {index}
Title: {metadata['title']}
Category: {metadata['category']}
Source: {metadata['source']}

Content:
{document['content']}
"""
        )

    context = "\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    # Oldingi suhbatni modelga uzatamiz
    if chat_history:
        messages.extend(chat_history)

    messages.append({
        "role": "user",
        "content": f"""
CONTEXT:

{context}

USER QUESTION:
{question}

Faqat context va suhbat tarixidagi ma'lumotlardan foydalan.
Agar javob uchun ma'lumot yetarli bo'lmasa, to'qima.
"""
    })

    answer = ask_gemma(messages)

    if "yetarli ma'lumot yo'q" in answer.lower():
        return NO_ANSWER

    return answer
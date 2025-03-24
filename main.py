import os
from dotenv import load_dotenv
import json
import sqlite3
import numpy as np
import faiss
import telebot
from sentence_transformers import SentenceTransformer
from langchain.chat_models.gigachat import GigaChat
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from database import create_database


# Загружаем переменные из .env файла
load_dotenv()

# Получаем токены из переменных окружения
sber_token = os.getenv("SBER_TOKEN")
bot_token = os.getenv("BOT_TOKEN")

# Инициализация бота и модели
bot = telebot.TeleBot(bot_token)
llm = GigaChat(credentials=sber_token, verify_ssl_certs=False)
create_database()

# Инициализация модели для формирования эмбеддингов параметров
embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def fetch_products():
    """
    Подключается к базе данных и извлекает список всех продуктов.

    Returns:
        list: Список всех продуктов с их характеристиками.
    """
    conn = sqlite3.connect("skincare_products.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, skin_type, effect, components, price FROM products")
    products = cursor.fetchall()
    conn.close()
    return products


def create_vector_store():
    """
    Создает векторное хранилище для продуктов на основе их описаний.

    Returns:
        tuple: Векторный индекс и список продуктов.
    """
    products = fetch_products()
    descriptions = [
        f"{name}, тип кожи: {skin}, эффект: {effect}, состав: {components}"
        for name, skin, effect, components, _ in products
    ]

    vectors = embedding_model.encode(descriptions, convert_to_numpy=True)

    # Создание FAISS-индекса
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    return index, products


# Загружаем базу в векторное хранилище
vector_index, product_list = create_vector_store()

# Словарь для хранения параметров пользователей
user_params = {}

# Загружаем модель для преобразования текста в векторное представление
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Подключаемся к базе данных и делаем запрос
conn = sqlite3.connect("skincare_products.db")
cursor = conn.cursor()
cursor.execute("SELECT name, skin_type, effect, components, price FROM products")
products = cursor.fetchall()

# Формируем текстовые описания продуктов для их векторизации
product_texts = [
    f"{name}. Подходит для {skin_type} кожи. Эффект: {effect}. Состав: {components}. Цена: {price} руб."
    for name, skin_type, effect, components, price in products
]

# Преобразуем текстовые описания продуктов в векторные представления
product_vectors = np.array([model.encode(text) for text in product_texts])

# Создаем FAISS индекс для поиска ближайших соседей в векторном пространстве
faiss_index = faiss.IndexFlatL2(product_vectors.shape[1])

# Добавляем векторные представления продуктов в индекс для дальнейшего поиска
faiss_index.add(product_vectors)

# Основной промпт
template = """
Ты — консультант-косметолог. Узнай у человека его предпочтения: тип кожи, желаемый эффект и ограничения (бюджет и аллергии).
Если он спрашивает что-то не по теме, говори, что это мне не интересно.

Текущий разговор:
{history}
Human: {input}
AI:
"""

# Создаем объект, отвечающий за историю общения
conversation = ConversationChain(
    llm=llm, verbose=True, memory=ConversationBufferMemory()
)
conversation.prompt.template = template


def get_recommendations(skin_type, effect, allergies, budget):
    """
    Подбирает товары по запросу пользователя, основываясь на типе кожи, желаемом эффекте, аллергиях и бюджете.

    Args:
        skin_type (str): Тип кожи пользователя.
        effect (str): Желаемый эффект от ухода.
        allergies (list): Список аллергенов, которых нужно избегать.
        budget (float): Максимальный бюджет на покупку.

    Returns:
        list: Список подходящих товаров с их характеристиками.
    """
    if allergies is None:
        allergies = []

    query_text = f"{skin_type} кожа, эффект: {effect}, бюджет: {budget}"
    query_vector = model.encode(query_text).reshape(1, -1)

    _, indices = faiss_index.search(query_vector, 10)
    recommendations = []
    total_price = 0

    for idx in indices[0]:
        name, prod_skin, prod_effect, components, price = products[idx]

        if any(allergy in components for allergy in allergies):
            continue
        if budget and total_price + price > budget:
            break
        recommendations.append((name, prod_skin, prod_effect, components, price))
        total_price += price
        if len(recommendations) >= 5:
            break

    return recommendations


def generate_explanation(products, skin_type, effect, allergies, budget):
    """
    Генерирует объяснение, почему выбраны именно эти товары, основываясь на параметрах пользователя.

    Args:
        products (list): Список выбранных товаров.
        skin_type (str): Тип кожи пользователя.
        effect (str): Желаемый эффект.
        allergies (list): Список аллергенов.
        budget (float): Максимальный бюджет.

    Returns:
        str: Объяснение выбора товаров с профессиональными советами.
    """
    product_list = "\n".join(
        [
            f"{name} (эффект: {effect}, цена: {price} руб.)"
            for name, _, effect, _, price in products
        ]
    )

    # Учитываем аллергии, если они есть
    allergy_text = (
        f" Пользователь избегает следующих компонентов из-за аллергии: {', '.join(allergies)}."
        if allergies
        else ""
    )

    # Учитываем бюджет, если он задан
    budget_text = f" Бюджет пользователя: до {budget} руб." if budget else ""

    # Формируем подробный промпт
    prompt = (
        f"Пользователь ищет уход за кожей с типом {skin_type}, желаемый эффект — {effect}.{allergy_text}{budget_text} "
        f"Вот подходящие товары:\n{product_list}\n\n"
        "Объясни, почему именно эти товары подходят пользователю, учитывая его пожелания и ограничения. "
        "Дай профессиональные советы по их применению."
    )

    explanation = llm.predict(input=prompt)
    return explanation


def extract_parameters_with_llm(text):
    """
    Извлекает ключевые параметры (тип кожи, эффект, ограничения) из текста с помощью LLM.

    Args:
        text (str): Текстовое сообщение пользователя.

    Returns:
        dict: Словарь с извлеченными параметрами (skin_type, effect, restrictions).
    """
    extraction_prompt = (
        "Выдели ключевые параметры: 1) тип кожи, 2) эффект, 3) ограничения (бюджет, аллергии). "
        "Верни только JSON с ключами 'skin_type', 'effect' и 'restrictions'. "
        f'Запрос: "{text}"'
    )
    extraction_response = llm.predict(extraction_prompt)
    try:
        return json.loads(extraction_response.strip("```json").strip("```"))
    except Exception as e:
        print(f"Ошибка при извлечении параметров: {e}")
        return None


def update_user_params(user_id, new_params):
    """
    Обновляет параметры пользователя, такие как тип кожи, желаемый эффект и ограничения (аллергии, бюджет).

    Args:
        user_id (str): Идентификатор пользователя.
        new_params (dict): Новые параметры, полученные от пользователя.
    """
    if user_id not in user_params:
        user_params[user_id] = {"skin_type": None, "effect": None, "restrictions": {}}

    # Обновляем skin_type и effect, если они присутствуют в новых параметрах
    if new_params.get("skin_type"):
        user_params[user_id]["skin_type"] = new_params["skin_type"]
    if new_params.get("effect"):
        user_params[user_id]["effect"] = new_params["effect"]

    # Обновляем ограничения, сохраняя уже установленные значения
    if isinstance(new_params.get("restrictions"), dict):
        new_restrictions = new_params["restrictions"]
        for key, value in new_restrictions.items():
            # Если значение не None, то обновляем
            if value is not None:
                # Для аллергий, если уже есть список, объединяем его с новым
                if key == "allergies":
                    current_allergies = user_params[user_id]["restrictions"].get(
                        "allergies", []
                    )
                    # Если новое значение – список, объединяем списки без дублирования
                    if isinstance(value, list):
                        combined = list(set(current_allergies + value))
                        user_params[user_id]["restrictions"]["allergies"] = combined
                    else:
                        # Если не список, просто записываем новое значение
                        user_params[user_id]["restrictions"]["allergies"] = value
                else:
                    # Для остальных ключей (например, budget) просто обновляем
                    user_params[user_id]["restrictions"][key] = value


@bot.message_handler(commands=["start"])
def send_welcome(message):
    """
    Отправляет приветственное сообщение при начале работы с ботом.

    Args:
        message (telebot.types.Message): Сообщение от пользователя с командой /start.
    """
    welcome_text = (
        "Добро пожаловать! Я ваш косметолог-ассистент. "
        "Расскажите, пожалуйста, о своих предпочтениях: тип кожи, желаемый эффект и ограничения (бюджет, аллергии)."
    )
    bot.send_message(message.chat.id, welcome_text)


@bot.message_handler(content_types=["text"])
def handle_text_message(message):
    """
    Обрабатывает текстовые сообщения от пользователя. Извлекает параметры и генерирует рекомендации по уходу.

    Args:
        message (telebot.types.Message): Текстовое сообщение от пользователя.
    """
    user_id = message.chat.id

    # Создаем или получаем историю общения для пользователя
    if user_id not in user_params:
        conversation.memory.chat_memory = []
    else:
        pass

    # Извлекаем сущности из ответа пользователя и заносим их в параметры
    params = extract_parameters_with_llm(message.text)
    if not params:
        bot.send_message(user_id, "Не могу распознать ваш запрос. Пожалуйста, уточните.")
        return

    # Обновляем параметры пользователя
    update_user_params(user_id, params)

    # Формируем рекомендации и объяснение
    recommendations = get_recommendations(
        user_params[user_id]["skin_type"],
        user_params[user_id]["effect"],
        user_params[user_id]["restrictions"].get("allergies", []),
        user_params[user_id]["restrictions"].get("budget"),
    )

    explanation = generate_explanation(
        recommendations,
        user_params[user_id]["skin_type"],
        user_params[user_id]["effect"],
        user_params[user_id]["restrictions"].get("allergies", []),
        user_params[user_id]["restrictions"].get("budget"),
    )

    # Отправляем рекомендации и объяснение пользователю
    bot.send_message(user_id, explanation)
    for i in range(0, len(explanation), 4000):
        bot.send_message(user_id, explanation[i:i + 4000])


if __name__ == "__main__":
    bot.polling(none_stop=True)

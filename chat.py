import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

# Загрузка данных и модели
with open('faq_data.json', 'r', encoding='utf-8') as f:
    faq_data = json.load(f)['faq_data']

questions = [item['question'] for item in faq_data]
answers = [item['answer'] for item in faq_data]

model = tf.keras.models.load_model('model/faq_model.keras')

# Загружаем preprocessor и bert отдельно
preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4")

# Функция для получения эмбеддингов
def get_embeddings(text):
    # Используем preprocessor для подготовки текста
    text_preprocessed = preprocessor(tf.constant([text]))
    # Получаем эмбеддинги через bert_encoder
    return bert_encoder(text_preprocessed)['pooled_output']

# Функция для поиска ответа
def find_answer(question):
    q_emb = get_embeddings(question).numpy()[0]
    max_score = -1
    best_answer = ""
    
    for answer in answers:
        a_emb = get_embeddings(answer).numpy()[0]
        # Объединяем эмбеддинги вопроса и ответа
        input_vector = np.concatenate([q_emb, a_emb])
        score = model.predict(input_vector.reshape(1, -1), verbose=0)[0][0]
        
        if score > max_score:
            max_score = score
            best_answer = answer
    
    return best_answer

# Основной цикл чата
print("Добро пожаловать в FAQ чат-бот! Введите 'выход' для завершения.")
while True:
    user_input = input("Вы: ")
    if user_input.lower() == 'выход':
        print("До свидания!")
        break
    
    try:
        answer = find_answer(user_input)
        print(f"Бот: {answer}")
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
        print("Попробуйте задать вопрос по-другому")

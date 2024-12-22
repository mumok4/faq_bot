import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

# Загрузка данных
with open('faq_data.json', 'r', encoding='utf-8') as f:
    faq_data = json.load(f)['faq_data']

questions = [item['question'] for item in faq_data]
answers = [item['answer'] for item in faq_data]

# Загрузка BERT модели и препроцессора
preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3")
bert_model = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4")

# Предобработка текста
def preprocess_text(text):
    return preprocessor(tf.constant([text]))

# Получение эмбеддингов
def get_embeddings(text):
    preprocessed = preprocess_text(text)
    return bert_model(preprocessed)['pooled_output']

# Создание датасета
dataset = []
for q, a in zip(questions, answers):
    q_emb = get_embeddings(q).numpy()
    a_emb = get_embeddings(a).numpy()
    dataset.append([q_emb[0], a_emb[0]])

dataset = np.array(dataset)

# Создание обучающих данных
X, Y = [], []
for i in range(dataset.shape[0]):
    for j in range(dataset.shape[0]):
        X.append(np.concatenate([dataset[i, 0, :], dataset[j, 1, :]], axis=0))
        Y.append(1 if i == j else 0)

X = np.array(X)
Y = np.array(Y)

# Создание модели
model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1536,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

# Компиляция и обучение модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, Y, epochs=100, batch_size=32, validation_split=0.2)

# Сохранение модели
model.save('model/faq_model.keras')

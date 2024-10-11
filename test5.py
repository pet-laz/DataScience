import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plot as plt
# Пример данных
diagnoses = np.array(['Грипп', 'Простуда', 'Пневмония', 'Тонзиллит', 
                      'Аллергия', 'Гастрит', 'Ротавирус', 'Ковид-19', 
                      'Синусит', 'Бронхит', 'Скарлатина', 'Менингит', 
                      'Миозит', 'Астма', 'Ринит', 'Гастроэнтерит', 'ОРВИ', 
                      'Краснуха', 'Энцефалит', 'Анорексия', 'Герпес', 'Аритмия', 
                      'Гайморит', 'Панкреатит', 'Эпилепсия', 'Гипертония', 'Стенокардия', 
                      'Анемия', 'Мигрень', 'Ветряная оспа', 'Инфаркт', 'Гипогликемия', 'Псориаз', 
                      'Язва желудка', 'Холецистит', 'Плеврит', 'Дискинезия', 'Демодекоз', 'Сепсис', 
                      'Гипотиреоз', 'Отит', 'Депрессия', 'Тромбоз', 'Гастроэзофагеальная рефлюксная болезнь',
                        'Пиелонефрит', 'Гепатит', 'Конъюнктивит', 'Ларингит'])
symptoms = np.array([['Кашель', 'Лихорадка', 'Головная боль'], 
                     ['Насморк', 'Кашель', 'Боль в горле'], 
                     ['Лихорадка', 'Кашель', 'Одышка'], 
                     ['Боль в горле', 'Озноб', 'Слабость'], 
                     ['Насморк', 'Головная боль', 'Чихание'], 
                     ['Тошнота', 'Рвота', 'Усталость'], 
                     ['Диарея', 'Боль в мышцах', 'Потливость'], 
                     ['Одышка', 'Слабость', 'Потеря вкуса'],
                       ['Заложенность носа', 'Потеря обоняния', 'Чихание'],
                         ['Кашель', 'Слабость', 'Потеря аппетита'], 
                         ['Сыпь', 'Потливость', 'Озноб'], ['Лихорадка', 'Тошнота', 'Слабость'],
                           ['Головная боль', 'Слабость', 'Боль в мышцах'], ['Кашель', 'Одышка', 'Слабость'],
                             ['Насморк', 'Потливость', 'Боль в горле'], ['Тошнота', 'Рвота', 'Диарея'], 
                             ['Лихорадка', 'Чихание', 'Заложенность носа'], 
                             ['Боль в горле', 'Сыпь', 'Потеря обоняния'], 
                             ['Головная боль', 'Тошнота', 'Потеря вкуса'],
                               ['Кашель', 'Озноб', 'Потеря аппетита'], 
 ['Сухость во рту', 'Головная боль', 'Зуд'], ['Боль в груди', 'Учащенное сердцебиение', 'Одышка'], 
 ['Заложенность носа', 'Головокружение', 'Усталость'], ['Снижение веса', 'Тошнота', 'Боль в животе'], 
 ['Слабость', 'Судороги', 'Головная боль'], ['Кашель', 'Лихорадка', 'Повышенная температура'], 
 ['Одышка', 'Учащенное сердцебиение', 'Сухость во рту'],
 ['Головная боль', 'Зуд', 'Потливость'], 
 ['Тошнота', 'Потеря аппетита', 'Боль в животе'], 
 ['Диарея', 'Озноб', 'Усталость'], 
 ['Кашель', 'Боль в груди', 'Потеря вкуса'], ['Лихорадка', 'Боль в горле', 'Одышка'],
   ['Сухость во рту', 'Слабость', 'Зуд'], ['Головная боль', 'Снижение веса', 'Тошнота'], 
   ['Одышка', 'Боль в груди', 'Учащенное сердцебиение'], ['Заложенность носа', 'Головная боль', 'Сухость во рту'], 
   ['Головокружение', 'Потеря вкуса', 'Боль в мышцах'], ['Тошнота', 'Судороги', 'Слабость'], ['Боль в животе', 'Диарея', 'Озноб'], 
   ['Зуд', 'Покраснение глаз', 'Потливость'], ['Повышенная температура', 'Лихорадка', 'Головная боль'], 
   ['Сыпь', 'Потливость', 'Зуд'], ['Головокружение', 'Потеря аппетита', 'Снижение веса'], 
   ['Боль в горле', 'Судороги', 'Головная боль'], ['Заложенность носа', 'Сухость во рту', 'Головная боль'], 
   ['Озноб', 'Сыпь', 'Тошнота'], ['Потливость', 'Снижение веса', 'Покраснение глаз'], 
   ['Головокружение', 'Боль в мышцах', 'Заложенность носа']])

# Кодируем диагнозы
label_encoder_diagnoses = LabelEncoder()
encoded_diagnoses = label_encoder_diagnoses.fit_transform(diagnoses)

# Кодируем симптомы
# Сначала создадим словарь для соответствия симптомов с номерами
all_symptoms = set(symptom for sublist in symptoms for symptom in sublist)
label_encoder_symptoms = LabelEncoder()
label_encoder_symptoms.fit(list(all_symptoms))

# Преобразуем симптомы в числовые значения
encoded_symptoms = [label_encoder_symptoms.transform(s) for s in symptoms]

# Паддинг для выравнивания длины массивов симптомов
max_length = max(len(s) for s in encoded_symptoms)
padded_symptoms = pad_sequences(encoded_symptoms, maxlen=max_length, padding='post')

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(padded_symptoms, encoded_diagnoses, test_size=0.2, random_state=42)

# Создаем модель нейронной сети
model = keras.Sequential([
    keras.layers.Input(shape=(max_length,)),
    keras.layers.Embedding(input_dim=len(label_encoder_symptoms.classes_), output_dim=8),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(len(label_encoder_diagnoses.classes_), activation='softmax')
])

# Компилируем модель
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Обучаем модель
model.fit(X_train, y_train, epochs=50)

# Оцениваем модель
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Тестовая точность: {accuracy}')

# Пример предсказания
sample_symptoms = np.array([['Кашель', 'Лихорадка', 'Головная боль']])
encoded_sample_symptoms = [label_encoder_symptoms.transform(s) for s in sample_symptoms]
padded_sample_symptoms = pad_sequences(encoded_sample_symptoms, maxlen=max_length, padding='post')

predicted_diagnosis = model.predict(padded_sample_symptoms)
predicted_diagnosis_label = label_encoder_diagnoses.inverse_transform([np.argmax(predicted_diagnosis)])

print(f'Предсказанный диагноз: {predicted_diagnosis_label[0]}')
index=np.where(diagnoses=="Простуда")[0][0]
print(index)
print(symptoms[index])
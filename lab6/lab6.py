import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup as bs
import tensorflow as tf
from tensorflow import keras

# ------------------------------------- сегмент API ------------------------------------
url = 'https://www.weather25.com/europe/ukraine'  # інтернет-джерело
observation_period = 'February'  # період спостереження
period_to_compare = ['March', 'April', 'May']  # період передбачення


# ------------------------------------- Завантаження даних ------------------------------------
# Функція, що повертає показники середньої температури днів для певного місяця
# Вхідні параметри:
#   response - http відповідь запиту
def get_days_temperature_from_month_page(response):
    if response.ok:
        result = []
        soup = bs(response.text, 'lxml')
        table = soup.find('table', class_='calendar_table')
        for day_info in table.tbody.find_all('div', class_='calendar_day_degree'):
            min_degree = day_info.find('span', class_='min-degree')
            max_degree = day_info.find('span', class_='max-degree')
            min_value = float(min_degree.find('span').contents[0])
            max_value = float(max_degree.find('span').contents[0])
            average_temperature = (min_value + max_value) / 2.0
            result.append(average_temperature)
        return result
    else:
        raise Exception("Проблеми зі з'єднанням")


# Функція, що повертає дані спостереження
# Вхідні параметри:
#   source_url - базовий url інтернет-джерела
#   month - місяць, температуру днів якого треба отримати
def get_input_dataset(source_url, month):
    full_url = source_url + '?page=month&month=' + month
    response = requests.get(full_url)
    return get_days_temperature_from_month_page(response)


# Функція, що повертає реальні дані для проміжку прогнозування
# Вхідні параметри:
#   source_url - базовий url інтернет-джерела
#   months - місяці, температуру днів яких треба отримати
def get_comparing_dataset(source_url, months):
    result = []
    for month in months:
        full_url = source_url + '?page=month&month=' + month
        response = requests.get(full_url)
        month_temperature = get_days_temperature_from_month_page(response)
        result += month_temperature
    return result


input_data = get_input_dataset(url, observation_period)
data_to_compare = get_comparing_dataset(url, period_to_compare)
real_data = input_data + data_to_compare

# Візуалізація даних періоду спостереження
plt.plot(input_data, label="Дані періоду спостереження")
plt.xlabel("День")
plt.ylabel("Середня температура дня")
plt.legend()
plt.show()

# Візуалізація реальних даних періоду спостереження та періоду прогнозу
plt.plot(real_data, label="Реальні дані для порівняння")
plt.xlabel("День")
plt.ylabel("Середня температура дня")
plt.legend()
plt.show()


# ------------------------------------- Прогноз за допомогою МНК ------------------------------------
# Функція, що повертає вектор виміряних даних
# Вхідні параметри:
#   arr - масив вхідних даних
def get_vector_of_measured_data(arr):
    n = len(arr)
    result = np.zeros((n, 1))
    for i in range(n):
        result[i, 0] = arr[i]
    return result


# Функція, що повертає матрицю базисних функцій
# Вхідні параметри:
#   n - розмір матриці
def get_matrix_of_basic_function_values(n):
    result = np.ones((n, 4))
    for i in range(n):
        result[i, 1] = float(i)
        result[i, 2] = float(i * i)
        result[i, 3] = float(i * i * i)
    return result


# Функція, що реалізує метод найменших квадратів
# Вхідні параметри:
#   arr - масив вхідних даних
def least_squares(arr):
    n = len(arr)
    Y = get_vector_of_measured_data(arr)
    F = get_matrix_of_basic_function_values(n)
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Y)
    result = F.dot(C)
    return [x[0] for x in result]


# Функція, що екстраполює за допомогою методу найменших квадратів
# Вхідні параметри:
#   arr - масив вхідних даних
#   forecast_value - величина на, яку необхідно зробити прогноз
def least_squares_extrapolation(arr, forecast_value):
    n = len(arr)
    Y = get_vector_of_measured_data(arr)
    F = get_matrix_of_basic_function_values(n)
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Y)
    j = n + 1
    predicted_values = []
    for i in range(0, forecast_value):
        predicted_values.append(C[0, 0] + C[1, 0] * j+ C[2, 0] * j * j+ C[3, 0] * j * j * j)
        j = j + 1
    return predicted_values


trend = least_squares(input_data)
MNK_prediction = trend + least_squares_extrapolation(input_data, len(data_to_compare))

# Візуалізація треду
plt.plot(input_data, label="Дані періоду спостереження")
plt.plot(trend, label="МНК тренд")
plt.xlabel("День")
plt.ylabel("Середня температура дня")
plt.legend()
plt.show()

# Візуалізація результатів прогнозу
plt.plot(real_data, label="Реальні дані зміни середньої температури дня")
plt.plot(MNK_prediction, label="Прогноз МНК")
plt.xlabel("День")
plt.ylabel("Середня температура дня")
plt.legend()
plt.show()

# ------------------------------------- Прогноз за допомогою нейромережі ------------------------------------
n = len(input_data)
X = np.zeros((n, 1), float)
for i in range(n):
    X[i, 0] = i

Y = np.zeros((n, 1), float)
for i in range(n):
    Y[i, 0] = input_data[i]

model = keras.Sequential([
    keras.layers.Input(shape=(1,), name="InputLayer"),
    keras.layers.Dense(100, activation='relu', name="HiddenLayer"),
    keras.layers.Dense(1, activation='linear', name="OutputLayer")
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

model.fit(X, Y, epochs=300, validation_split=0.1)

ml_predicted = []
for x in range(len(real_data)):
    value = model.predict([[x]])
    ml_predicted.append(value[0])

# Візуалізація результатів прогнозу
plt.plot(real_data, label="Реальні дані зміни середньої температури дня")
plt.plot(ml_predicted, label="Прогноз нейромережі")
plt.xlabel("День")
plt.ylabel("Середня температура дня")
plt.legend()
plt.show()

# ------------------------------------- Порівняння прогнозів МНК та нейромережі ------------------------------------
# Обчислення статистичних характеристик
m_MNK_prediction = np.median(MNK_prediction)
d_MNK_prediction = np.var(MNK_prediction)
scv_MNK_prediction = math.sqrt(d_MNK_prediction)
print('----- статистичні характеристики МНК прогнозу  -----')
print('математичне сподівання ВВ =', m_MNK_prediction)
print('дисперсія ВВ =', d_MNK_prediction)
print('СКВ ВВ =', scv_MNK_prediction)
print('-----------------------------------------------------------------------')

m_ml_predicted = np.median(ml_predicted)
d_ml_predicted = np.var(ml_predicted)
scv_ml_predicted = math.sqrt(d_ml_predicted)
print('----- статистичні характеристики пронозу нейромережі  -----')
print('математичне сподівання ВВ =', m_ml_predicted)
print('дисперсія ВВ =', d_ml_predicted)
print('СКВ ВВ =', scv_ml_predicted)
print('-----------------------------------------------------------------------')

m_real_data = np.median(real_data)
d_real_data = np.var(real_data)
scv_real_data = math.sqrt(d_real_data)
print('----- статистичні характеристики реальних даних  -----')
print('математичне сподівання ВВ =', m_real_data)
print('дисперсія ВВ =', d_real_data)
print('СКВ ВВ =', scv_real_data)
print('-----------------------------------------------------------------------')

# Порівняльний графік
plt.plot(real_data, label="Реальні дані зміни середньої температури дня")
plt.plot(MNK_prediction, label="Прогноз МНК")
plt.plot(ml_predicted, label="Прогноз нейромережі")
plt.xlabel("День")
plt.ylabel("Середня температура дня")
plt.legend()
plt.show()

# Порівняння результатів прогнозу за різними методами
start_date = datetime.datetime(2022, 2, 1)
for i in range(len(input_data), len(real_data)):
    date = start_date + datetime.timedelta(days=i)
    print(date.date(), " прогноз МНК: ", MNK_prediction[i], " прогноз нейромережі: ", ml_predicted[i][0],
          " реальні дані ", real_data[i])

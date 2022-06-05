import math
import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup as bs

# ------------------------------------- сегмент API ------------------------------------
base_url = 'https://index.minfin.com.ua/ua/reference/coronavirus/ukraine/hmelnickaya'   # Посилання на інтернет-джерело
start_year, start_month = 2020, 3                   # початкова дата відстеження коронавірусу в регіоні
end_year, end_month = 2022, 1                       # кінцева дата відстеження коронавірусу в регіоні
forecast_by_days, forecast_by_months = 182, 6       # значення передбачення


# ------------------------------------- Завантаження даних ------------------------------------
# Функція, що формує дату необхідного формату з року та місяця
# Вхідні параметри:
#   year - рік
#   month - місяць
def get_formatted_date(year, month):
    return str(year) + '-' + str(month).rjust(2, '0')


# Функція, що формує повний url до даних за місяць
# Вхідні параметри:
#   url - базовий url
#   date - дата
def create_full_url(url, date):
    return url + '/' + date + '/'


# Функція, повертає дані зі html тексту сторінки
# Вхідні параметри:
#   response - відповідь http запиту
#   date - дата
def get_values_from_response(response, date):
    if response.ok:
        soup = bs(response.text, 'lxml')
        tables = [div.table for div in soup.find_all('div',  class_='compact-table')]
        table = tables[-1]
        day_infos = table.find_all('tr')
        for day_info in day_infos:
            if day_info.find('th'):
                continue
            day_info = day_info.find_all('td', align='right')
            for i in range(3):
                found = day_info[i].contents[0]
                value = 0
                if found.name != 'br':
                    value = int(found)
                data[i].append((date, value))
    else:
        print("Проблеми зі з'єднанням")


data = [[], [], []]     # масив, що містить дані з сайту
session = requests.Session()       # перевикористання сесії для збільшення швидкості завантаження даних
while start_year < end_year or (start_year == end_year and start_month <= end_month):
    formatted_date = get_formatted_date(start_year, start_month)
    full_url = create_full_url(base_url, formatted_date)
    request_response = session.get(full_url)
    get_values_from_response(request_response, formatted_date)
    if start_month != 12:
        start_month += 1
    else:
        start_month = 1
        start_year += 1

# Формування даних по днях
infected_per_day = np.array([value for date, value in data[0]])
dead_per_day = np.array([value for date, value in data[1]])
recovered_per_day = np.array([value for date, value in data[2]])


# ------------------------------------- Візуалізація отриманих даних ------------------------------------
plt.plot(infected_per_day, label="Кількість інфікованих за день", color="blue")
plt.legend()
plt.show()

plt.plot(dead_per_day, label="Смертельні випадки за день", color="black")
plt.legend()
plt.show()

plt.plot(recovered_per_day, label="Видужали за день", color="green")
plt.legend()
plt.show()


# ------------------------------------- Сегментація даних ------------------------------------
# Функція, що сегментує дані по місяцях
# Вхідні параметри:
#   arr - структура даних, що містить дані та дати місяців
def segment_by_month(arr):
    d = {}
    for date, value in arr:
        if date not in d:
            d[date] = 0
        d[date] += value
    return list(d.values())


# Формування даних по місяцях
infected_per_month = np.array(segment_by_month(data[0]))
dead_per_month = np.array(segment_by_month(data[1]))
recovered_per_month = np.array(segment_by_month(data[2]))

# ------------------------------------- Візуалізація сегментованих даних ------------------------------------
plt.plot(infected_per_month, label="Всього інфіковано за місяць", color="blue")
plt.legend()
plt.show()

plt.plot(dead_per_month, label="Смертельні випадки за місяць", color="black")
plt.legend()
plt.show()

plt.plot(recovered_per_month, label="Видужали за місяць", color="green")
plt.legend()
plt.show()

# ------------------------------------- Обчислення статистичних характеристик ------------------------------------
m_infected_per_day = np.median(infected_per_day)
d_infected_per_day = np.var(infected_per_day)
scv_infected_per_day = math.sqrt(d_infected_per_day)
print('----- статистичні характеристики вибірки "Кількість інфікованих за день"  -----')
print('математичне сподівання ВВ =', m_infected_per_day)
print('дисперсія ВВ =', d_infected_per_day)
print('СКВ ВВ =', scv_infected_per_day)
print('-----------------------------------------------------------------------')

m_dead_per_day = np.median(dead_per_day)
d_dead_per_day = np.var(dead_per_day)
scv_dead_per_day = math.sqrt(d_dead_per_day)
print('----- статистичні характеристики вибірки "Смертельні випадки за день"  -----')
print('математичне сподівання ВВ =', m_dead_per_day)
print('дисперсія ВВ =', d_dead_per_day)
print('СКВ ВВ =', scv_dead_per_day)
print('-----------------------------------------------------------------------')

m_recovered_per_day = np.median(recovered_per_day)
d_recovered_per_day = np.var(recovered_per_day)
scv_recovered_per_day = math.sqrt(d_recovered_per_day)
print('----- статистичні характеристики вибірки "Видужали за день"  -----')
print('математичне сподівання ВВ =', m_recovered_per_day)
print('дисперсія ВВ =', d_recovered_per_day)
print('СКВ ВВ =', scv_recovered_per_day)
print('-----------------------------------------------------------------------')

m_infected_per_month = np.median(infected_per_month)
d_infected_per_month = np.var(infected_per_month)
scv_infected_per_month = math.sqrt(d_infected_per_month)
print('----- статистичні характеристики вибірки "Всього інфіковано за місяць"  -----')
print('математичне сподівання ВВ =', m_infected_per_month)
print('дисперсія ВВ =', d_infected_per_month)
print('СКВ ВВ =', scv_infected_per_month)
print('-----------------------------------------------------------------------')

m_dead_per_month = np.median(dead_per_month)
d_dead_per_month = np.var(dead_per_month)
scv_dead_per_month = math.sqrt(d_dead_per_month)
print('----- статистичні характеристики вибірки "Смертельні випадки за місяць"  -----')
print('математичне сподівання ВВ =', m_dead_per_month)
print('дисперсія ВВ =', d_dead_per_month)
print('СКВ ВВ =', scv_dead_per_month)
print('-----------------------------------------------------------------------')

m_recovered_per_month = np.median(recovered_per_month)
d_recovered_per_month = np.var(recovered_per_month)
scv_recovered_per_month = math.sqrt(d_recovered_per_month)
print('----- статистичні характеристики вибірки "Видужали за місяць"  -----')
print('математичне сподівання ВВ =', m_recovered_per_month)
print('дисперсія ВВ =', d_recovered_per_month)
print('СКВ ВВ =', scv_recovered_per_month)
print('-----------------------------------------------------------------------')


# ------------------------------------- Визначення трендів за допомогою МНК ------------------------------------
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
    result = np.ones((n, 5))
    for i in range(n):
        result[i, 1] = float(i)
        result[i, 2] = float(i * i)
        result[i, 3] = float(i * i * i)
        result[i, 4] = float(i * i * i * i)
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


# Формування трендів
infected_per_day_trend = least_squares(infected_per_day)
dead_per_day_trend = least_squares(dead_per_day)
recovered_per_day_trend = least_squares(recovered_per_day)
infected_per_month_trend = least_squares(infected_per_month)
dead_per_month_trend = least_squares(dead_per_month)
recovered_per_month_trend = least_squares(recovered_per_month)

# Візуалізація трендів
plt.plot(infected_per_day, label="Кількість інфікованих за день", color="blue")
plt.plot(infected_per_day_trend, label="МНК тренд", color="orange")
plt.legend()
plt.show()

plt.plot(dead_per_day, label="Смертельні випадки за день", color="black")
plt.plot(dead_per_day_trend, label="МНК тренд", color="orange")
plt.legend()
plt.show()

plt.plot(recovered_per_day, label="Видужали за день", color="green")
plt.plot(recovered_per_day_trend, label="МНК тренд", color="orange")
plt.legend()
plt.show()

plt.plot(infected_per_month, label="Всього інфіковано за місяць", color="blue")
plt.plot(infected_per_month_trend, label="МНК тренд", color="orange")
plt.legend()
plt.show()

plt.plot(dead_per_month, label="Смертельні випадки за місяць", color="black")
plt.plot(dead_per_month_trend, label="МНК тренд", color="orange")
plt.legend()
plt.show()

plt.plot(recovered_per_month, label="Видужали за місяць", color="green")
plt.plot(recovered_per_month_trend, label="МНК тренд", color="orange")
plt.legend()
plt.show()


# ------------------------------------- Екстраполяція за допомогою МНК ------------------------------------
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
        predicted_values.append(C[0, 0] + C[1, 0] * j + (C[2, 0] * j * j) + (C[3, 0] * j * j * j) +
                                (C[4, 0] * j * j * j * j))
        j = j + 1
    return predicted_values


# Формування результату прогнозу
infected_per_day_forecast = least_squares_extrapolation(infected_per_day, forecast_by_days)
dead_per_day_forecast = least_squares_extrapolation(dead_per_day, forecast_by_days)
recovered_per_day_forecast = least_squares_extrapolation(recovered_per_day, forecast_by_days)
infected_per_month_forecast = least_squares_extrapolation(infected_per_month, forecast_by_months)
dead_per_month_forecast = least_squares_extrapolation(dead_per_month, forecast_by_months)
recovered_per_month_forecast = least_squares_extrapolation(recovered_per_month, forecast_by_months)

# Візуалізація прогнозів
plt.plot(infected_per_day_forecast, label="Кількість інфікованих за день прогноз", color="orange")
plt.legend()
plt.show()

plt.plot(dead_per_day_forecast, label="Смертельні випадки за день прогноз", color="orange")
plt.legend()
plt.show()

plt.plot(recovered_per_day_forecast, label="Видужали за день прогноз", color="orange")
plt.legend()
plt.show()

plt.plot(infected_per_month_forecast, label="Всього інфіковано за місяць прогноз", color="orange")
plt.legend()
plt.show()

plt.plot(dead_per_month_forecast, label="Смертельні випадки за місяць прогноз", color="orange")
plt.legend()
plt.show()

plt.plot(recovered_per_month_forecast, label="Видужали за місяць прогноз", color="orange")
plt.legend()
plt.show()

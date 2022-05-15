import numpy as np
import pandas as pd

# ------------------------------------- сегмент API ------------------------------------
# Шлях до файлу з вхідними даними
input_data_file_path = ".//data.xlsx"
elements = ["Приватбанк", "Укргазбанк", "Ощадбанк", "Креді Агріколь Банк",     # Масив, що містить назви колонок елементів дослідження
            "Акордбанк", "Прокредит Банк", "ОТП Банк", "Таскомбанк"]

criteria_column = "Критерій"                                # Назва колонки, яка визначає природу критерію фактора

G = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]        # Масив, що містить вагові коефіцієнти факторів
G_group = [1, 1, 1, 1]                                      # Масив, що містить вагові коефіцієнти груп факторів

d = 0.001                                                   # Відносний коефіцієнт запасу

factors_count = len(G)                                      # Кількість факторів
group_count = len(G_group)                                  # Кількість елементів
elements_count = len(elements)                              # Кількість груп факторів


# Функція, що приймає на вхід масив та нормалізує його
def normalize_array(arr):
    arr_length = len(arr)
    arr_sum = sum(arr)
    res = [arr[k]/arr_sum for k in range(arr_length)]
    return np.array(res)


# ------------------------------------- Зчитування даних з файлу ------------------------------------
factors = np.zeros((factors_count, elements_count), float)          # Масив зі значеннями факторів
input_data = pd.read_excel(input_data_file_path, engine='openpyxl')
for i in range(factors_count):
    for j in range(elements_count):
        factors[i, j] = input_data[elements[j]][i]      # Зчитуємо дані з файлу в масив

criteria = {}                                           # Словник, що визначає критерій кожного фактору
for i in range(factors_count):
    criteria[i] = input_data[criteria_column][i]

# ------------------------ Власне алгоритм вирішення багатокритеріальної задачі -----------------------
# -------------------------- за критерієм згортки (використовується -----------------------------------
# -------------------------- нелінійа схема компромісів А.М. Вороніна) --------------------------------

G_normalized = normalize_array(G)                       # Нормалізовані вагові коефіцієнти факторів
G_group_normalized = normalize_array(G_group)           # Нормалізовані вагові коефіцієнти груп факторів

# Нормуємо значення критеріїв, що входять до згортки
factors_normalized = np.zeros((factors_count, elements_count), float)
for i in range(factors_count):
    factor = factors[i]
    if criteria[i] == 'min':
        max_value = max(factor)
        error = max(d * max_value, d)       # будуємо масштабоване допустиме відхилення
        for j in range(elements_count):
            factors_normalized[i, j] = factor[j] / (max_value + error)
    else:
        min_value = min(factor)
        error = max(d * min_value, d)       # будуємо масштабоване допустиме відхилення
        for j in range(elements_count):
            factors_normalized[i, j] = (min_value - error) / factor[j]

# Проводимо зведення часткових факторів до узагальнених за групою
factors_by_group = np.zeros((group_count, elements_count), float)
for i in range(elements_count):
    # Формуємо значення першої групи (рейтингові)
    for j in range(0, 3):
        factors_by_group[0, i] += G_normalized[j] * (1 - factors_normalized[j, i]) ** (-1)

    # Формуємо значення другої групи (робота з сайтом)
    for j in range(3, 7):
        factors_by_group[1, i] += G_normalized[j] * (1 - factors_normalized[j, i]) ** (-1)

    # Формуємо значення третьої групи (технічні)
    for j in range(7, 12):
        factors_by_group[2, i] += G_normalized[j] * (1 - factors_normalized[j, i]) ** (-1)

    # Формуємо значення четвертої групи (економічні)
    for j in range(12, 16):
        factors_by_group[3, i] += G_normalized[j] * (1 - factors_normalized[j, i]) ** (-1)

# Нормуємо частинні показники факторів для узагальненої оцінки
factors_normalized_maximum = np.zeros((factors_count, elements_count), float)
for i in range(factors_count):
    factor = factors[i]
    if criteria[i] == 'min':
        for j in range(elements_count):
            error = max(d * factor[j], d)  # будуємо масштабоване допустиме відхилення
            factors_normalized_maximum[i, j] = factor[j] / (factor[j] + error)
    else:
        for j in range(elements_count):
            error = max(d * factor[j], d)  # будуємо масштабоване допустиме відхилення
            factors_normalized_maximum[i, j] = (factor[j] - error) / factor[j]

# Формуємо з отриманих значень масив, що містить вже максимальні значення для груп факторів
factors_by_group_maximum = np.zeros((group_count, elements_count), float)       # Масив, що містить нормовані
for i in range(elements_count):
    # Формуємо значення першої групи
    for j in range(0, 3):
        factors_by_group_maximum[0, i] += G_normalized[j] * (1 - factors_normalized_maximum[j, i]) ** (-1)

    # Формуємо значення другої групи
    for j in range(3, 7):
        factors_by_group_maximum[1, i] += G_normalized[j] * (1 - factors_normalized_maximum[j, i]) ** (-1)

    # Формуємо значення третьої групи
    for j in range(7, 12):
        factors_by_group_maximum[2, i] += G_normalized[j] * (1 - factors_normalized_maximum[j, i]) ** (-1)

    # Формуємо значення четвертої групи
    for j in range(12, 16):
        factors_by_group_maximum[3, i] += G_normalized[j] * (1 - factors_normalized_maximum[j, i]) ** (-1)

# Розраховуємо узагальнену нормовану оцінку для груп факторів
factors_by_group_normalized = np.zeros((group_count, elements_count), float)
for i in range(group_count):
    for j in range(elements_count):
        factors_by_group_normalized[i, j] = factors_by_group[i, j] / factors_by_group_maximum[i, j]

# Отримуємо масив зі значеннями інтегрованої оцінки ефективності досліджуваних елементів
result = np.zeros(elements_count, float)
for i in range(elements_count):
    for j in range(group_count):
        result[i] += G_group_normalized[j] * (1 - factors_by_group_normalized[j, i]) ** (-1)

# ------------------------------ Інтерпретація результату ------------------------------------
# Приведення значення до єдиної шкали (від 0 до 1)
maximum_group_values = []    # Найгірші значення з можливих для груп факторів
for i in range(group_count):
    maximum_group_values.append((1 - max(factors_by_group_normalized[i])) ** (-1))
max_res = sum(maximum_group_values)

result_in_single_scale = np.zeros(elements_count, float)    # Масив, що містить рішення багатокритеріальної
for i in range(elements_count):                             # задачі в єдиній шкалі
    result_in_single_scale[i] = 1 - result[i] / max_res

# Пошук оптимального варіанту
opt, opt_idx = 0, 0
for i in range(group_count):
    if opt < result_in_single_scale[i]:
        opt = result_in_single_scale[i]
        opt_idx = i

print("Reslut = ", result_in_single_scale)
print("Номер оптимального варіанту:", opt_idx+1)
print("Власне оптимальний варіант:", elements[opt_idx])

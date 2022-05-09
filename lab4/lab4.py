import numpy as np
import pandas as pd

input_data_file_path = "/home/koroliuk/PycharmProjects/DataScience-labs/lab4/data.xlsx"
elements = ["Сайт1", "Сайт2", "Сайт3", "Сайт4", "Сайт5", "Сайт6", "Сайт7", "Сайт8", "Сайт9", "Сайт10", "Сайт11",
            "Сайт12"]
criteria_column = "Критерій"
G = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
d = 0.1

factors_count = len(G)
elements_count = len(elements)
G_normalized = np.zeros(factors_count)
G_sum = sum(G)
for i in range(factors_count):
    G_normalized[i] = G[i] / G_sum

factors = np.zeros((factors_count, elements_count),  float)
input_data = pd.read_excel(input_data_file_path, engine='openpyxl')
for i in range(factors_count):
    for j in range(elements_count):
        factors[i, j] = input_data[elements[j]][i]

criteria = {}
for i in range(factors_count):
    criteria[i] = input_data[criteria_column][i]

factors_normalized = np.zeros((factors_count, elements_count),  float)
for i in range(factors_count):
    factor = factors[i]
    if criteria[i] == 'min':
        max_value = max(factor)
        for j in range(elements_count):
            factors_normalized[i, j] = factor[j]/(max_value+d)
    else:
        min_value = min(factor)
        for j in range(elements_count):
            factors_normalized[i, j] = (min_value-d)/factor[j]

factors_by_group = np.zeros((4, elements_count), float)
for i in range(elements_count):
    # 1st group
    for j in range(0, 3):
        factors_by_group[0, i] += (1 - factors_normalized[j, i])**(-1)

    # 2nd group
    for j in range(3, 10):
        factors_by_group[1, i] += (1 - factors_normalized[j, i]) ** (-1)

    # 3rd group
    for j in range(10, 12):
        factors_by_group[2, i] += (1 - factors_normalized[j, i]) ** (-1)

    # 4th group
    for j in range(12, 16):
        factors_by_group[3, i] += (1 - factors_normalized[j, i]) ** (-1)

factors_normalized_maximum = np.zeros((factors_count, elements_count),  float)
for i in range(factors_count):
    factor = factors[i]
    if criteria[i] == 'min':
        max_value = max(factor)
        for j in range(elements_count):
            factors_normalized_maximum[i, j] = factor[j] / (factor[j] + d)
    else:
        min_value = min(factor)
        for j in range(elements_count):
            factors_normalized_maximum[i, j] = (factor[j] - d) / factor[j]

factors_by_group_maximum = np.zeros((4, elements_count), float)
for i in range(elements_count):
    # 1st group
    for j in range(0, 3):
        factors_by_group_maximum[0, i] += (1 - factors_normalized_maximum[j, i])**(-1)

    # 2nd group
    for j in range(3, 10):
        factors_by_group_maximum[1, i] += (1 - factors_normalized_maximum[j, i]) ** (-1)

    # 3rd group
    for j in range(10, 12):
        factors_by_group_maximum[2, i] += (1 - factors_normalized_maximum[j, i])**(-1)

    # 4th group
    for j in range(12, 16):
        factors_by_group_maximum[3, i] += (1 - factors_normalized_maximum[j, i])**(-1)

factors_by_group_normalized = np.zeros((4, elements_count), float)
for i in range(4):
    for j in range(elements_count):
        factors_by_group_normalized[i, j] = factors_by_group[i, j]/factors_by_group_maximum[i, j]

integro = np.zeros(elements_count, float)
for i in range(elements_count):
    for j in range(4):
        integro[i] += (1-factors_by_group_normalized[j, i])**(-1)

max_values = []
for i in range(4):
    max_values.append((1-max(factors_by_group_normalized[i]))**(-1))
max_integro = sum(max_values)
integro_normalized = np.zeros(elements_count, float)
for i in range(elements_count):
    integro_normalized[i] = 1 - integro[i]/max_integro

print(integro_normalized)

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------- сегмент API ------------------------------------
file_path = './Data_Set_9.xlsx'
forecast_value = 20

# ------------------------------------- Завантаження даних ------------------------------------
df = pd.read_excel(file_path, engine='openpyxl')

# Візуалізація завантажених даних
print(df)

# ------------------------------------- Розрахунок економічних показників ------------------------------------
n = len(df['Account'])
order_details = []
for i in range(n):
    quantity = int(df['Quantity'][i])
    price = float(df['Price'][i])
    status = str(df['Status'][i])
    order_details.append({
        'quantity': quantity,
        'price': price,
        'status': status
    })

sales = []
profit = []
for order_detail in order_details:
    sale = order_detail['quantity'] * order_detail['price']
    status = order_detail['status']

    sales.append(sale)
    if status == 'declined':
        profit.append(0.)
    else:
        profit.append(sale)

# Візуалізація отриманих даних
plt.plot(sales, label="Продажі")
plt.legend()
plt.show()

plt.plot(profit, label="Прибуток")
plt.legend()
plt.show()

# ------------------------------------- Розрахунок статистичних характеристик ------------------------------------
m_sales = np.median(sales)
d_sales = np.var(sales)
scv_sales = math.sqrt(d_sales)
print('----- статистичні характеристики вибірки продаж  -----')
print('математичне сподівання ВВ =', m_sales)
print('дисперсія ВВ =', d_sales)
print('СКВ ВВ =', scv_sales)
print('-----------------------------------------------------------------------')

m_profit = np.median(profit)
d_profit = np.var(profit)
scv_profit = math.sqrt(d_profit)
print('----- статистичні характеристики вибірки прибутку  -----')
print('математичне сподівання ВВ =', m_profit)
print('дисперсія ВВ =', d_profit)
print('СКВ ВВ =', scv_profit)
print('-----------------------------------------------------------------------')


# ------------------------------------- Побудова тренду за МНК ------------------------------------
# Функція, що реалізує метод найменших квадратів
# Вхідні параметри:
#   arr - масив вхідних даних
#   degree - порядок поліному
def least_squares(arr, degree):
    n = len(arr)
    arr_x = [_ for _ in range(n)]
    fitted_pn = np.polyfit(arr_x, arr, degree)
    return np.polyval(fitted_pn, arr_x)


sales_trend = least_squares(sales, 2)
profit_trend = least_squares(profit, 2)

# Візуалізація отриманих даних
plt.plot(sales, label="Продажі")
plt.plot(sales_trend, label="МНК тренд")
plt.legend()
plt.show()

plt.plot(profit, label="Прибуток")
plt.plot(profit_trend, label="МНК тренд")
plt.legend()
plt.show()


# ------------------------------------- Формування прогнозу ------------------------------------
# Функція, що екстраполює за допомогою методу найменших квадратів
# Вхідні параметри:
#   arr - масив вхідних даних
#   forecast_value - величина на, яку необхідно зробити прогноз
#   degree - порядок поліному
def least_squares_extrapolation(arr, forecast, degree):
    n = len(arr)
    arr_x = [_ for _ in range(n)]
    fitted_pn = np.polyfit(arr_x, arr, degree)

    forecast_x = [_ for _ in range(n, n + forecast)]
    total_x = arr_x + forecast_x
    return np.polyval(fitted_pn, total_x)


sales_prediction = least_squares_extrapolation(sales, forecast_value, 2)
profit_prediction = least_squares_extrapolation(profit, forecast_value, 2)

# Візуалізація отриманих даних
# Графіком
plt.plot(sales, label="Продажі")
plt.plot(sales_prediction, label="Прогноз")
plt.legend()
plt.show()

plt.plot(profit, label="Прибуток")
plt.plot(profit_prediction, label="Прогноз")
plt.legend()
plt.show()

# Табличкою
print('------------------------------- Прогноз для продаж -----------------------------')
print('--- номер часового інтервалу ------- значення --- ')
for i in range(len(sales), len(sales_prediction)):
    print(f'--- {i} ------- {sales_prediction[i]} --- ')
print('-------------------------------------------------------------------------------------------------------')

print('------------------------------- Прогноз для прибутку -----------------------------')
print('--- номер часового інтервалу ------- значення --- ')
for i in range(len(profit), len(profit_prediction)):
    print(f'--- {i} ------- {profit_prediction[i]} --- ')
print('-------------------------------------------------------------------------------------------------------')

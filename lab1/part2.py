import math

import numpy
import matplotlib.pyplot as plt

# модель випадкової величини - похибки вимірювання

n = int(10000)  # кількість реалізацій випадкової величини,
# що забезпечує ітераційність реалізації методу Монте-Карло

# закон розподілу випадкової похибки - рівномірний
# генерація випадкової похибки в масив error із систематикою offset
offset = -7
error = numpy.random.rand(n)*20  # коректура параметрів закону розподілу
error += offset

uniform = numpy.random.rand(n) * 20  # рівномірний розподіл від [0,20) без систематики для порівняння

# гістограма закону розподілу випадкової похибки
plt.hist(error, bins=20, facecolor="blue", alpha=0.5)
plt.show()

# модель досліджуваного процесу
# закон розподілу - лінійний
process = numpy.zeros(n)
for i in range(n):
    process[i] = 0.008 * i

# графік моделі досліджуваного процесу
plt.plot(process)
plt.show()

# Адитивна модель експериментальних даних
# Лінійний процес з рівномірним шумом
total = numpy.zeros(n)
for i in range(n):
     total[i] = process[i] + error[i]

# Графік адитивної моделі
plt.plot(total)
plt.plot(process)
plt.show()

# гістограми законів розподілу експериментальних даних
plt.hist(error, bins=20, alpha=0.5, label='error')
plt.hist(uniform, bins=20, alpha=0.5, label='test')
plt.hist(total, bins=20, alpha=0.5, label='total')
plt.show()

# статистичні характеристики трендової вибірки (зміщені)
print("Cтатистичні характеристики трендової вибірки (зміщені):")
m_total = numpy.median(total)
d_total = numpy.var(total)
scv_total = math.sqrt(d_total)
print('матиматичне сподівання ВВ3=', m_total)
print('дисперсія ВВ3 =', d_total)
print('СКВ ВВ3=', scv_total)

# оцінка статистичних характеристик ВВ з урахуванням динаміки зміни контрольованої велечини
calc_error1 = numpy.zeros(n)
for i in range(n):
    calc_error1[i] = total[i] - process[i]

print("Статистичні характеристики випадкової похибки:")
m_calc_error1 = numpy.median(calc_error1)
d_calc_error1 = numpy.var(calc_error1)
scv_calc_error1 = math.sqrt(d_calc_error1)
print('матиматичне сподівання ВВ =', m_calc_error1)
print('дисперсія ВВ =', d_calc_error1)
print('СКВ ВВ =', scv_calc_error1)

# оцінка статистичних характеристик ВВ з урахуванням динаміки зміни контрольованої велечини та без систематики
calc_error2 = numpy.zeros(n)
for i in range(n):
    calc_error2[i] = total[i] - process[i] - offset

# Графік адитивної моделі, без помилки систематики
plt.plot(calc_error2 + process)
plt.plot(process)
plt.show()

# гістограми законів розподілу експериментальних даних
plt.hist(uniform, bins=20, alpha=0.5, label='test')
plt.hist(calc_error2, bins=20, alpha=0.5, label='out2')
plt.hist(total, bins=20, alpha=0.5, label='total')
plt.show()

# статистичні характеристики випадкової похибки без систематики
print("Статистичні характеристики випадкової похибки без систематики:")
m_calc_error2 = numpy.median(calc_error2)
d_calc_error2 = numpy.var(calc_error2)
scv_calc_error2 = math.sqrt(d_calc_error2)
print('матиматичне сподівання ВВ без систематики =', m_calc_error2)
print('дисперсія ВВ без систематики =', d_calc_error2)
print('СКВ ВВ без систематики =', scv_calc_error2)

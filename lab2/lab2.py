import math
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------- сегмент API -----------------------------------
n = 1000                                    # кількість реалізацій ВВ
dm = 0; dsig = 5                            # параметри нормального закону розподілу випадкової похибки: середнє та СКВ
anomalies_percentage = 10                           # кількість АВ у відсотках та абсолютних одиницях
anomalies = int((n * anomalies_percentage)/100)

t = 0.02                                   # дискретність (темп) оновлення інформації

# ------------------------------ Модель випадкової похибки -------------------------
errors = np.random.normal(dm, dsig, n)      # має нормальний закон розподілу за умовою

# ----------------------------------- Модель тренду --------------------------------
trend = np.zeros(n)                         # квадратична модель реального процесу
for i in range(n):
    trend[i] = 0.00005 * i * i

# --------------------------- Модель вимірів (зашумлена без АВ) ---------------------
s = np.zeros(n)
for i in range(n):
    s[i] = trend[i] + errors[i]

# ------------------------------- Модель аномальних вимірів -------------------------
anomalies_positions = np.zeros(anomalies)
for i in range(anomalies):
    anomalies_positions[i] = np.random.randint(0, n)    # рівномірний розкид номерів АВ в межах вибірки розміром 0-n

m_av_pos = np.median(anomalies_positions)
d_av_pos = np.var(anomalies_positions)
scv_av_pos = math.sqrt(d_av_pos)
print('номери АВ: anomalies_positions = ', anomalies_positions)
print('----- статистичні характеристики РІВНОМІРНОГО закону розподілу ВВ -----')
print('математичне сподівання ВВ =', m_av_pos)
print('дисперсія ВВ =', d_av_pos)
print('СКВ ВВ =', scv_av_pos)
print('-----------------------------------------------------------------------')

plt.hist(anomalies_positions, bins=10, facecolor="blue", alpha=0.5)
plt.show()

anomalies_errors = np.random.normal(dm, 3 * dsig, anomalies)    # генерація аномальних значень помилки, які розподілені
                                                                # за нормальним законом згідно умови, параметри: dm, 3*dsig

# --------------------------- Модель вимірів (зашумлена з АВ) ---------------------
s_av = s.copy()
for i in range(anomalies):
    k = int(anomalies_positions[i])
    s_av[k] = s[k] + anomalies_errors[i]


# ---------------------------- Виявлення аномальних вимірів -----------------------
# Функція, що будує функцію для обчислення
# Вхідні параметри:
#   a - інтенсивність зміни досліджуваного процесу
def build_func(a_coef):
    # Функція, що вираховує значення лівої частини рівняння для виявлення АВ за коефіцієнтом старіння
    # Вхідні параметри:
    #   s - швидкість старіння інформації
    def f(s_opt):
        return (s_opt + 4) * ((s_opt - 1) ** 5) - a_coef * ((s_opt + 4) ** 4)
    return f


# Функція, що знаходить корінь рівняння func = 0 за методом дихотомії
#   l - ліва границя пошуку
#   r - права границя пошуку
#   e - точність
def solve_equation(l, r, func, e):
    a, b = l, r
    ya = func(a)
    yb = func(b)
    if ya * yb > 0:
        return None
    while b - a > e:
        x = (b + a) / 2
        ya = func(a)
        yx = func(x)
        if ya * yx == 0:
            break
        elif ya * yx > 0:
            a = x
        else:
            b = x
    return (b + a) / 2


# Функція, яка перевіряє чи вимір є аномальним
def is_anomaly(a_coef):
    func = build_func(a_coef)
    root = solve_equation(0, 2, func, 0.000001)
    if root is None:
        return True
    if 0 <= root <= 2:
        return False
    return True


# Функція, що обчислює інтенсивність зміни досліджуваного процесу
def calculate_a_coef(pos):
    s_av_k = s_av[:pos + 1]
    alpha_k = s_av[pos] - s_av[pos - 1]     # друга кінцева різниця
    ds_k = np.var(s_av_k)                   # дисперсія вимірювання
    return t * alpha_k / ds_k


detected_anomalies_positions = [False] * n
# Вважатимемо перший вимір неаномальним, так як для перевірки необхідно мати попереднє значення
for k in range(1, n):
    a_k = calculate_a_coef(k)
    detected_anomalies_positions[k] = is_anomaly(a_k)      # визначаємо чи є вимів аномальним

# ---------------------------- Усунення аномальних вимірів -----------------------
s_without_av = []               # вибірка, в якій аномальні виміри відкинуті
trend_without_av_v = []         # тренд, в якому відкинуті виміри, що стосуються аномальних
for i in range(n):
    if detected_anomalies_positions[i] is False:
        s_without_av.append(s_av[i])
        trend_without_av_v.append(trend[i])

# ----------------------------------- МНК згладження -----------------------------
# Функція обчислень алгоритму - MNK
def MNK(Yin, F):
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    Yout = F.dot(C)
    return Yout


Yin = np.zeros((len(s_without_av), 1))
F = np.ones((len(s_without_av), 3))
for i in range(len(s_without_av)):  # формування структури вхідних матриць МНК
    Yin[i, 0] = float(s_without_av[i])  # формування матриці вхідних даних без аномалій
    F[i, 1] = float(i)
    F[i, 2] = float(i * i)  # формування матриці вхідних даних без аномалій

# застосування МНК до зашумлених вимірів з усунутими аномаліями
Yout = MNK(Yin, F)

# Графік зашумленої вибірки без АВ
plt.plot(s)
plt.plot(trend)
plt.show()

# Графік зашумленої вибірки з АВ
plt.plot(s_av)
plt.plot(trend)
plt.show()

# ------------------------- Обчислення статистичних характеристик -------------------
# Статистичні характеристики нормальної похибки вимірів
m_err = np.median(errors)
d_err = np.var(errors)
scv_err = math.sqrt(d_err)
print('------- Статистичні характеристики випадкової НОРМАЛЬНОЇ похибки вимірів -----')
print('математичне сподівання ВВ =', m_err)
print('дисперсія ВВ =', d_err)
print('СКВ ВВ =', scv_err)
print('------------------------------------------------------------------')

plt.hist(errors, bins=20, facecolor="blue", alpha=0.5)
plt.show()

# Статистичні характеристики вхідної вибірки значень (зашумленої без аномальних вимірів)
# з урахуванням тренду
s0 = np.zeros(n)
for i in range(n):
    s0[i] = s[i] - trend[i]     # урахування тренду в оцінках статистичних характеристик

m_s0 = np.median(s0)
d_s0 = np.var(s0)
scv_s0 = math.sqrt(d_s0)
print('------- Статистичні характеристики НОРМАЛЬНОЇ похибки вимірів -----')
print('------- вхідної вибірки значень (зашумленої без аномальних вимірів) -----')
print('математичне сподівання ВВ=', m_s0)
print('дисперсія ВВ =', d_s0)
print('СКВ ВВ=', scv_s0)
print('------------------------------------------------------------------')

plt.hist(s0, bins=20, facecolor="blue", alpha=0.5)
plt.show()

# Статистичні характеристики аномальної вибірки (зашумленої з аномальними вимірами)
# з урахуванням тренду
s1 = np.zeros(n)
for i in range(n):
    s1[i] = s_av[i] - trend[i]     # урахування тренду в оцінках статистичних характеристик

m_s1 = np.median(s1)
d_s1 = np.var(s1)
scv_s1 = math.sqrt(d_s1)
print('------- Статистичні характеристики НОРМАЛЬНОЇ похибки вимірів -----')
print('-------  аномальної вибірки (зашумленої з аномальними вимірами) -----')
print('математичне сподівання ВВ=', m_s1)
print('дисперсія ВВ =', d_s1)
print('СКВ ВВ=', scv_s1)
print('------------------------------------------------------------------')

plt.hist(s1, bins=20, facecolor="blue", alpha=0.5)
plt.show()

# Статистичні характеристики результатів згладжування МНК
s2 = np.zeros(len(s_without_av))
for i in range(len(s_without_av)):
    s2[i] = Yout[i, 0] - trend_without_av_v[i]  # урахування тренду в оцінках статистичних характеристик

m_s2 = np.median(s2)
d_s2 = np.var(s2)
scv_s2 = math.sqrt(d_s2)
print('------- Статистичні характеристики НОРМАЛЬНОЇ похибки вимірів -----')
print('------- результатів згладжування МНК -----')
print('математичне сподівання ВВ=', m_s2)
print('дисперсія ВВ =', d_s2)
print('СКВ ВВ=', scv_s2)
print('------------------------------------------------------------------')

plt.hist(s2, bins=20, facecolor="blue", alpha=0.5)
plt.show()

# Табличкою
print('------------------------------- Статистичні характеристики вибірок --------------------1---------')
print('----------------------------- похибки нормальні ------- похибки аномальні --- згладження МНК ---')
print('математичне сподівання ВВ3 =', m_s0,   '----', m_s1,  '----', m_s2)
print('дисперсія ВВ3 =            ', d_s0,  '----', d_s1,  '----', d_s2)
print('СКВ ВВ3 =                   ', scv_s0,'----', scv_s1,'----', scv_s2)
print('-------------------------------------------------------------------------------------------------------')

# Графіки (в одному графічному вікні): квадратичного тренду; зашумленої без аномальних вимірів вибірки; зашумленої
# з аномальними вимірами вибірки; результатів згладжування МНК.
plt.plot(s_av)
plt.plot(s)
plt.plot(trend)
plt.plot(Yout)
plt.show()

# Гістограми (в одному графічному вікні) похибок: зашумленої без
# аномальних вимірів вибірки; зашумленої з аномальними вимірами вибірки; результатів згладжування МНК.
plt.hist(errors, bins=20, alpha=0.5, label='S')
plt.hist(s0, bins=20, alpha=0.5, label='S1')
plt.hist(s1, bins=20, alpha=0.5, label='S2')
plt.hist(s2, bins=20, alpha=0.5, label='S3')
plt.show()

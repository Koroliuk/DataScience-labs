import math
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------- сегмент API ------------------------------------
n = 100                                      # кількість реалізацій ВВ
dm, dsig = 0, 30                            # параметри нормального закону розподілу випадкової похибки: середнє та СКВ
anomalies_percentage = 10                           # кількість АВ у відсотках та абсолютних одиницях
anomalies = int((n * anomalies_percentage)/100)

t = 1                                        # дискретність (темп) оновлення інформації

# ------------------------------ Модель випадкової похибки ------------------------------
errors = np.random.normal(dm, dsig, n)       # має нормальний закон розподілу за умовою

# -------------------------------------- Модель тренду ----------------------------------
trend = np.zeros(n)
for i in range(n):
    trend[i] = 0.05 * i * i                  # квадратична модель реального процесу

# ----------------------------- Модель вимірів (зашумлена без АВ) -----------------------
s = np.zeros(n)
for i in range(n):
    s[i] = trend[i] + errors[i]

# --------------------------------- Модель аномальних вимірів ---------------------------
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

# ----------------------------- Модель вимірів (зашумлена з АВ) -------------------------
s_av = s.copy()
for i in range(anomalies):
    k = int(anomalies_positions[i])
    s_av[k] = s[k] + anomalies_errors[i]

# ------------------------------- Виявлення аномальних вимірів ---------------------------
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
    if ya == 0:
        return a
    if yb == 0:
        return b
    if ya * yb > 0:
        return None
    x = 0
    while b - a > e:
        dx = (b - a) / 2
        x = a + dx
        ya = func(a)
        yx = func(x)
        if ya * yx < 0:
            b = x
        else:
            a = x
    return x


# Функція, яка перевіряє чи вимір є аномальним
def is_anomaly(a_coef):
    func = build_func(a_coef)
    root = solve_equation(0, 2, func, 0.0001)
    if root is None:
        return True
    if 0 <= root <= 2:
        return False
    return True


# Функція, що обчислює інтенсивність зміни досліджуваного процесу
def calculate_a_coef(pos):
    alpha_k = (s_av[pos] - s_av[pos - 1])/t - (s_av[pos-1] - s_av[pos - 2])/t      # друга кінцева різниця
    ds_k = np.var(s_av)                                                            # дисперсія вимірювання
    return t * alpha_k / ds_k


detected_anomalies_positions = [False] * n
# Вважатимемо перші два виміри неаномальним, так як для перевірки необхідно мати попередні значення
for k in range(2, n):
    a_k = calculate_a_coef(k)
    detected_anomalies_positions[k] = is_anomaly(a_k)      # визначаємо чи є вимів аномальним

# ---------------------------- Усунення аномальних вимірів -----------------------
s_without_av = []               # вибірка, в якій аномальні виміри відкинуті
trend_without_av_v = []         # тренд, в якому відкинуті виміри, що стосуються аномальних
for i in range(n):
    if detected_anomalies_positions[i] is False:
        s_without_av.append(s_av[i])
        trend_without_av_v.append(trend[i])

# ---------- Згладжування матричним фільтром Калмана другого порядку --------------
# Функція, що проводить екстраполяцію вектора стану системи
# Вхідні параметри:
#   X - оцінка стану процесу
#   P - коваріаційна матриця процесу
#   F - матриця еволюції процесу
# Повертає:
#   ex_X - екстраполяцію (передбачення) вектора стану процесу
#   ex_P - коваріаційну матрицю для екстрапольованого значення вектора стану
def kf_predict(X, P, F):
    ex_X = np.dot(F, X)
    ex_P = np.dot(np.dot(F, P), F.T)
    return ex_X, ex_P


# Функція, що корегує екстрапольоване значення
# Вхідні параметри:
#   X - екстраполяція (передбачення) вектора стану процесу
#   P - коваріаційна матриця для екстрапольованого значення вектора стану
#   z - істинний вектор стану процесу (виміри)
#   H - матриця вимірів
#   R - коваріаційна матриця шуму вимірів
# Повертає:
#   X - скорегований вектор стану процесу
#   P - скорегована коваріаційна матриця
def kf_update(X, P, z, H, R):
    PHt = np.dot(P, H.T)
    HPHt = np.dot(np.dot(H, P), H.T)
    HPHt_R = np.linalg.inv(HPHt+R)
    HP = np.dot(H, P)
    P = P - np.dot(np.dot(PHt, HPHt_R), HP)
    X = X + np.dot(np.dot(np.dot(P, H.T), np.linalg.inv(R)), z-X)
    return X, P


# Матриця процесу для моделі другого порядку
F = np.array([[1, t, (t ** 2) / 2],
              [0, 1.0, t],
              [0, 0, 1.0]])


v_without_av = np.array([s_without_av[i] - s_without_av[i-1] for i in range(1, len(s_without_av))])
a_without_av = np.array([v_without_av[i] - v_without_av[i-1] for i in range(1, len(v_without_av))])
x_var = np.var(s_without_av)
v_var = np.var(v_without_av)
a_var = np.var(a_without_av)
# Коваріаційна матриця стану - визначає "впевненість" фільтра в оцінці змінних стану
P = np.array([[x_var, 0, 0],
              [0, v_var, 0],
              [0, 0, a_var]])
# Матриця вимірів відповідно до моделі другого порядку
H = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])

# Коваріаційна матриця помилки вимірів
R = np.array([[x_var, 0, 0],
              [0, v_var, 0],
              [0, 0, a_var]])

# Початковий стан
X = np.array([[s_without_av[2]],
              [(s_without_av[2] - s_without_av[1]) / t],
              [(((s_without_av[2] - s_without_av[1]) / t) - ((s_without_av[1] - s_without_av[0]) / t)) / t]])
print(len(s_without_av))
filtered_result = [s_without_av[0], s_without_av[1]]
for i in range(2, len(s_without_av)):
    Z = np.array([
        [s_without_av[i]],
        [(s_without_av[i] - s_without_av[i - 1]) / t],
        [(((s_without_av[i] - s_without_av[i - 1]) / t) - ((s_without_av[i - 1] - s_without_av[i - 2]) / t)) / t]
    ])

    ex_X, ex_P = kf_predict(X, P, F)
    X, P = kf_update(ex_X, ex_P, Z, H, R)
    filtered_result.append(X[0, 0])

filtered_result = np.array(filtered_result)

# ------------------- Обчислення статистичних характеристик закону -------------------------
# ----------------------- розподілу випадкової похибки вимірів -----------------------------

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

# Для вхідної вибірки значень (зашумлена без АВ)
s0 = np.zeros(n)
for i in range(n):
    s0[i] = s[i] - trend[i]     # урахування тренду

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

# Для аномальної вибірки (зашумлена з АВ)
s1 = np.zeros(n)
for i in range(n):
    s1[i] = s_av[i] - trend[i]     # урахування тренду

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

# Результати згладжування
s2 = np.zeros(len(filtered_result))
for i in range(len(filtered_result)):
    s2[i] = filtered_result[i] - trend_without_av_v[i]     # урахування тренду

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

# -------------------------- Відображення результатів розрахунків ------------------------------
# Табличкою
print('------------------------------- Статистичні характеристики вибірок --------------------1---------')
print('----------------------------- похибки нормальні ------- похибки аномальні --- згладження ---')
print('математичне сподівання ВВ3 =', m_s0,   '----', m_s1,  '----', m_s2)
print('дисперсія ВВ3 =            ', d_s0,  '----', d_s1,  '----', d_s2)
print('СКВ ВВ3 =                   ', scv_s0,'----', scv_s1,'----', scv_s2)
print('-------------------------------------------------------------------------------------------------------')

# Графіки
plt.plot(s_av, label="Зашумлена з АВ", color="#99AAFF")
plt.plot(s, label="Зашумлена без АВ", color="#A7F594")
plt.plot(filtered_result, label="Результати згладжування", color="#224411")
plt.plot(trend, label="Квадратичний тренд", color="#FF6633")
plt.legend()
plt.show()

# Гістограми
plt.hist(s1, bins=20, label='Зашумлена з АВ', color="#99AAFF")
plt.hist(s0, bins=20, label='Зашумлена без АВ', color="#A7F594")
plt.hist(s2, bins=20, label='Результати згладження', color="#224411")
plt.legend()
plt.show()

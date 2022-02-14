# Варіант 9, високий рівень
import numpy
import matplotlib.pyplot as plt

n = int(10000)

# Модель випадкової величини - похибки вимірювання
# Закон зміни похибки - експоненційний

errors = numpy.random.exponential(size=n)-1
plt.hist(errors, bins=20, facecolor="blue", alpha=0.5)
# plt.hist(errors-20, bins=20, facecolor="green", alpha=0.5)

plt.show()

# Модель зміни досліджуваного процесу
# Закон зміни досліджуваного процесу - квадратичний

process = numpy.zeros(n)
for i in range(n):
    process[i] = 0.0000005 * i * i
plt.plot(process)
plt.show()

# Адитивна модель експериментальних даних

total = numpy.zeros(n)
for i in range(n):
    total[i] = errors[i] + process[i]

plt.plot(total)
plt.plot(process)
plt.show()

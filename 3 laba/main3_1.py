import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, lagrange

# Загрузка данных
data = pd.read_csv('DataSet3_1.csv')

# Преобразуем метки времени в формат datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Выбираем 10 точек, равномерно распределённых по датасету и создаем копию
num_points = 10
selected_points = data.iloc[::len(data)//num_points].copy()

# Преобразуем метки времени в числовой формат (ordinal)
selected_points['Timestamp_ordinal'] = selected_points['Timestamp'].apply(lambda x: x.toordinal())

# Получаем значения для интерполяции
x_values = selected_points['Timestamp_ordinal']
y_values = selected_points['Power (kW)']

# 1. Линейная интерполяция
# Линейная интерполяция заключается в проведении прямых линий между соседними точками данных.
# Она применяется, когда требуется простое и быстрое приближение данных, но не учитывает возможную кривизну между точками.
linear_interp = np.interp(np.linspace(x_values.min(), x_values.max(), 100), x_values, y_values)

# Выводим график линейной интерполяции
plt.figure(figsize=(10,6))
plt.plot(x_values, y_values, 'o', label='Исходные точки')
plt.plot(np.linspace(x_values.min(), x_values.max(), 100), linear_interp, '-', label='Линейная интерполяция')
plt.xlabel('Timestamp (ordinal)')
plt.ylabel('Power (kW)')
plt.title('Линейная интерполяция потребляемой мощности')
plt.legend()
plt.grid(True)
plt.show()

# 2. Кубический сплайн
# Кубический сплайн – это метод интерполяции, который использует кусочно-кубические полиномы для создания гладкой кривой через набор данных.
# В отличие от линейной интерполяции, кубический сплайн более плавно моделирует поведение между точками.
cubic_spline = CubicSpline(x_values, y_values)
x_new = np.linspace(x_values.min(), x_values.max(), 100)
y_cubic = cubic_spline(x_new)

# Выводим график кубического сплайна
plt.figure(figsize=(10,6))
plt.plot(x_values, y_values, 'o', label='Исходные точки')
plt.plot(x_new, y_cubic, '-', label='Кубический сплайн')
plt.xlabel('Timestamp (ordinal)')
plt.ylabel('Power (kW)')
plt.title('Интерполяция кубическим сплайном потребляемой мощности')
plt.legend()
plt.grid(True)
plt.show()

# 3. Интерполяция Лагранжа
# Полином Лагранжа используется для интерполяции набора точек с использованием полиномиальной функции.
# Основная идея заключается в том, что полином проходит через все заданные точки. Метод может стать нестабильным для больших наборов данных.
lagrange_poly = lagrange(x_values.values, y_values.values)
y_lagrange = np.polyval(lagrange_poly, x_new)

# Выводим график интерполяции Лагранжа
plt.figure(figsize=(10,6))
plt.plot(x_values, y_values, 'o', label='Исходные точки')
plt.plot(x_new, y_lagrange, '-', label='Интерполяция Лагранжа')
plt.xlabel('Timestamp (ordinal)')
plt.ylabel('Power (kW)')
plt.title('Интерполяция методом Лагранжа потребляемой мощности')
plt.legend()
plt.grid(True)
plt.show()

# 4. Интерполяция методом Ньютона
# Интерполяция Ньютона использует метод разделённых разностей для построения полинома.
# Она хорошо подходит для вычислений, когда точки могут быть добавлены или удалены из набора данных.
def divided_diff(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:,0] = y
    for j in range(1,n):
        for i in range(n-j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])
    return coef[0, :]

def newton_poly(coef, x_data, x):
    n = len(coef) - 1
    p = coef[n]
    for k in range(1, n + 1):
        p = coef[n-k] + (x - x_data[n-k])*p
    return p

# Интерполяция Ньютона
coef_newton = divided_diff(x_values.values, y_values.values)
y_newton = [newton_poly(coef_newton, x_values.values, xi) for xi in x_new]

# Выводим график интерполяции Ньютона
plt.figure(figsize=(10,6))
plt.plot(x_values, y_values, 'o', label='Исходные точки')
plt.plot(x_new, y_newton, '-', label='Интерполяция Ньютона')
plt.xlabel('Timestamp (ordinal)')
plt.ylabel('Power (kW)')
plt.title('Интерполяция методом Ньютона потребляемой мощности')
plt.legend()
plt.grid(True)
plt.show()

# Объединённый график всех методов интерполяции
plt.figure(figsize=(10,6))
plt.plot(x_values, y_values, 'o', label='Исходные точки')

# Линейная интерполяция
plt.plot(np.linspace(x_values.min(), x_values.max(), 100), linear_interp, '-', label='Линейная интерполяция')

# Кубический сплайн
plt.plot(x_new, y_cubic, '-', label='Кубический сплайн')

# Интерполяция Лагранжа
plt.plot(x_new, y_lagrange, '-', label='Интерполяция Лагранжа')

# Интерполяция Ньютона
plt.plot(x_new, y_newton, '-', label='Интерполяция Ньютона')

# Оформление графика
plt.xlabel('Timestamp (ordinal)')
plt.ylabel('Power (kW)')
plt.title('Сравнение различных методов интерполяции')
plt.legend()
plt.grid(True)
plt.show()

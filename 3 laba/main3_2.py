import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, lagrange

# Функции для интерполяции методом Ньютона
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

# Загрузка данных
data_2 = pd.read_csv('DataSet3_2.csv', delimiter=';')

# Преобразуем метки времени и числовые данные в правильный формат
data_2['Dollar'] = data_2['Dollar'].str.replace(',', '.').astype(float)
data_2['Oil Brent'] = data_2['Oil Brent'].str.replace(',', '.').astype(float)
data_2['Data'] = pd.to_datetime(data_2['Data'], format='%d.%m.%Y')

# Выбираем 10 точек, равномерно распределённых по датасету и создаем копию
num_points = 10
selected_points_2 = data_2.iloc[::len(data_2)//num_points].copy()

# Преобразуем метки времени в числовой формат (ordinal)
selected_points_2['Data_ordinal'] = selected_points_2['Data'].apply(lambda x: x.toordinal())

# Получаем значения для интерполяции (курс доллара)
x_values_2 = selected_points_2['Data_ordinal']
y_values_2 = selected_points_2['Dollar']

# 1. Линейная интерполяция
linear_interp_2 = np.interp(np.linspace(x_values_2.min(), x_values_2.max(), 100), x_values_2, y_values_2)

# Выводим график линейной интерполяции
plt.figure(figsize=(10,6))
plt.plot(x_values_2, y_values_2, 'o', label='Исходные точки')
plt.plot(np.linspace(x_values_2.min(), x_values_2.max(), 100), linear_interp_2, '-', label='Линейная интерполяция')
plt.xlabel('Data (ordinal)')
plt.ylabel('Dollar')
plt.title('Линейная интерполяция курса доллара')
plt.legend()
plt.grid(True)
plt.show()

# 2. Кубический сплайн
cubic_spline_2 = CubicSpline(x_values_2, y_values_2)
x_new_2 = np.linspace(x_values_2.min(), x_values_2.max(), 100)
y_cubic_2 = cubic_spline_2(x_new_2)

# Выводим график кубического сплайна
plt.figure(figsize=(10,6))
plt.plot(x_values_2, y_values_2, 'o', label='Исходные точки')
plt.plot(x_new_2, y_cubic_2, '-', label='Кубический сплайн')
plt.xlabel('Data (ordinal)')
plt.ylabel('Dollar')
plt.title('Интерполяция кубическим сплайном курса доллара')
plt.legend()
plt.grid(True)
plt.show()

# 3. Интерполяция Лагранжа
lagrange_poly_2 = lagrange(x_values_2.values, y_values_2.values)
y_lagrange_2 = np.polyval(lagrange_poly_2, x_new_2)

# Выводим график интерполяции Лагранжа
plt.figure(figsize=(10,6))
plt.plot(x_values_2, y_values_2, 'o', label='Исходные точки')
plt.plot(x_new_2, y_lagrange_2, '-', label='Интерполяция Лагранжа')
plt.xlabel('Data (ordinal)')
plt.ylabel('Dollar')
plt.title('Интерполяция методом Лагранжа курса доллара')
plt.legend()
plt.grid(True)
plt.show()

# 4. Интерполяция методом Ньютона
coef_newton_2 = divided_diff(x_values_2.values, y_values_2.values)
y_newton_2 = [newton_poly(coef_newton_2, x_values_2.values, xi) for xi in x_new_2]

# Выводим график интерполяции Ньютона
plt.figure(figsize=(10,6))
plt.plot(x_values_2, y_values_2, 'o', label='Исходные точки')
plt.plot(x_new_2, y_newton_2, '-', label='Интерполяция Ньютона')
plt.xlabel('Data (ordinal)')
plt.ylabel('Dollar')
plt.title('Интерполяция методом Ньютона курса доллара')
plt.legend()
plt.grid(True)
plt.show()

# 5. Объединённый график всех методов интерполяции
plt.figure(figsize=(10,6))
plt.plot(x_values_2, y_values_2, 'o', label='Исходные точки')

# Линейная интерполяция
plt.plot(np.linspace(x_values_2.min(), x_values_2.max(), 100), linear_interp_2, '-', label='Линейная интерполяция')

# Кубический сплайн
plt.plot(x_new_2, y_cubic_2, '-', label='Кубический сплайн')

# Интерполяция Лагранжа
plt.plot(x_new_2, y_lagrange_2, '-', label='Интерполяция Лагранжа')

# Интерполяция Ньютона
plt.plot(x_new_2, y_newton_2, '-', label='Интерполяция Ньютона')

# Оформление графика
plt.xlabel('Data (ordinal)')
plt.ylabel('Dollar')
plt.title('Сравнение различных методов интерполяции курса доллара')
plt.legend()
plt.grid(True)
plt.show()

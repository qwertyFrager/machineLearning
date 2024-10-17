import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных из CSV файла
data1 = pd.read_csv('DataSet1_3.csv', sep=',')

# Преобразуем столбец Date в формат datetime, чтобы с ним можно было работать как с временными рядами
data1["Date"] = pd.to_datetime(data1["Date"])

# Преобразуем дату в числовой формат: время в секундах с начала эпохи Unix
data1['Date'] = data1['Date'].astype(np.int64) / 1e9

# Определяем начало и конец диапазона данных для анализа
start_idx = 100
end_idx = 300

# Создаем новый DataFrame с основными признаками:
# 'x' — время (в секундах), сдвинутое так, чтобы отсчет начинался с 1
# 'y' — цена закрытия акций (целевой параметр)
# 'p1' — цена открытия акций
# 'p2' — цена закрытия акций с отставанием на один шаг назад
data = pd.DataFrame()
data['x'] = data1['Date'][start_idx:end_idx].to_numpy() - data1['Date'][start_idx] + 1
data['y'] = data1['Close'][start_idx:end_idx].to_numpy()
data['p1'] = data1['Open'][start_idx:end_idx].to_numpy()
data['p2'] = data1['Close'][start_idx-1:end_idx-1].to_numpy()

# Определяем размер обучающей выборки как 80% от общего количества данных
n = round(0.8 * (end_idx - start_idx))

# Разделение на обучающую (train) и тестовую (test) выборки
train = data[0:n]  # 80% данных
test = data[n:]    # оставшиеся 20% данных

# Визуализируем исходные данные
fig, ax = plt.subplots(figsize=(10, 6))

# Зеленый цвет — цена закрытия, красный — цена открытия, синий — сдвинутая цена закрытия (на один шаг назад)
ax.plot(train['x'], train['y'], color='green', linestyle='solid', label='Train Close')
ax.plot(train['x'], train['p1'], color='red', linestyle='solid', label='Train Open')
ax.plot(train['x'], train['p2'], color='blue', linestyle='solid', label='Train Lag Close')

ax.plot(test['x'], test['y'], color='green', linestyle='dashed', label='Test Close')
ax.plot(test['x'], test['p1'], color='red', linestyle='dashed', label='Test Open')
ax.plot(test['x'], test['p2'], color='blue', linestyle='dashed', label='Test Lag Close')

# Устанавливаем заголовок и подписи осей
ax.set(title="Исходные данные: тренировка и тест", xlabel="Время", ylabel="Цена акций")
ax.legend()
plt.show()

# Разделяем данные на признаки (x, p1, p2) и целевую переменную (y)
train_x = train.drop(['y'], axis=1, inplace=False)
train_y = train['y']
test_x = test.drop(['y'], axis=1, inplace=False)
test_y = test['y']

# МНК (Метод Наименьших Квадратов, Quadratic Hypothesis)
from scipy.optimize import curve_fit

# Определяем функцию для квадратичной модели
def mapping_func(x, a, b, c, b1, b2):
    return a + b * x['x'] + c * x['x']**2 + b1 * x['p1'] + b2 * x['p2']

# Применяем метод curve_fit для подбора параметров модели на обучающих данных
args, _ = curve_fit(mapping_func, train_x, train_y)

# Извлекаем коэффициенты модели
a, b, c, b1, b2 = args

# Прогнозы для обучающей и тестовой выборки
res_y_tr = a + b * train['x'] + c * train['x']**2 + b1 * train['p1'] + b2 * train['p2']
res_y_ts = a + b * test['x'] + c * test['x']**2 + b1 * test['p1'] + b2 * test['p2']

# Визуализация результатов МНК
fig, ax = plt.subplots(figsize=(10, 6))

# Зеленый цвет — реальные значения, красный — прогнозируемые значения
ax.plot(train['x'], train['y'], color='green', linestyle='solid', label='Train Real')
ax.plot(test['x'], test['y'], color='green', linestyle='dashed', label='Test Real')
ax.plot(train['x'], res_y_tr, color='red', linestyle='solid', label='Train Predicted')
ax.plot(test['x'], res_y_ts, color='red', linestyle='dashed', label='Test Predicted')

ax.set(title="Прогнозирование методом МНК", xlabel="Время", ylabel="Цена акций")
ax.legend()
plt.show()

# SVM (Support Vector Machine) — Метод опорных векторов для регрессии
from sklearn.svm import SVR

# Создаем и обучаем модель SVM с радиальным базисным ядром (RBF)
svm_model = SVR(kernel='rbf')
svm_model.fit(train_x, train_y)

# Прогнозы для обучающей и тестовой выборки с использованием SVM
res_y_tr_svm = svm_model.predict(train_x)
res_y_ts_svm = svm_model.predict(test_x)

# Визуализация результатов модели SVM
fig, ax = plt.subplots(figsize=(10, 6))

# Зеленый цвет — реальные значения, красный — прогнозируемые значения
ax.plot(train['x'], train['y'], color='green', linestyle='solid', label='Train Real')
ax.plot(test['x'], test['y'], color='green', linestyle='dashed', label='Test Real')
ax.plot(train['x'], res_y_tr_svm, color='red', linestyle='solid', label='Train Predicted SVM')
ax.plot(test['x'], res_y_ts_svm, color='red', linestyle='dashed', label='Test Predicted SVM')

ax.set(title="Прогнозирование методом SVM", xlabel="Время", ylabel="Цена акций")
ax.legend()
plt.show()

# Lasso (линейная регрессия с L1-регуляризацией)
from sklearn.linear_model import Lasso

# Создаем и обучаем модель Lasso с регуляризацией
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(train_x, train_y)

# Прогнозы для обучающей и тестовой выборки с использованием Lasso
res_y_tr_lasso = lasso_model.predict(train_x)
res_y_ts_lasso = lasso_model.predict(test_x)

# Визуализация результатов модели Lasso
fig, ax = plt.subplots(figsize=(10, 6))

# Зеленый цвет — реальные значения, красный — прогнозируемые значения
ax.plot(train['x'], train['y'], color='green', linestyle='solid', label='Train Real')
ax.plot(test['x'], test['y'], color='green', linestyle='dashed', label='Test Real')
ax.plot(train['x'], res_y_tr_lasso, color='red', linestyle='solid', label='Train Predicted Lasso')
ax.plot(test['x'], res_y_ts_lasso, color='red', linestyle='dashed', label='Test Predicted Lasso')

ax.set(title="Прогнозирование методом Lasso", xlabel="Время", ylabel="Цена акций")
ax.legend()
plt.show()

# PLS (Partial Least Squares) — метод частных наименьших квадратов
from sklearn.cross_decomposition import PLSRegression

# Создаем и обучаем модель PLS с 3 компонентами
pls_model = PLSRegression(n_components=3)
pls_model.fit(train_x, train_y)

# Прогнозы для обучающей и тестовой выборки с использованием PLS
res_y_tr_pls = pls_model.predict(train_x)
res_y_ts_pls = pls_model.predict(test_x)

# Визуализация результатов модели PLS
fig, ax = plt.subplots(figsize=(10, 6))

# Зеленый цвет — реальные значения, красный — прогнозируемые значения
ax.plot(train['x'], train['y'], color='green', linestyle='solid', label='Train Real')
ax.plot(test['x'], test['y'], color='green', linestyle='dashed', label='Test Real')
ax.plot(train['x'], res_y_tr_pls, color='red', linestyle='solid', label='Train Predicted PLS')
ax.plot(test['x'], res_y_ts_pls, color='red', linestyle='dashed', label='Test Predicted PLS')

ax.set(title="Прогнозирование методом PLS", xlabel="Время", ylabel="Цена акций")
ax.legend()
plt.show()



# SVM (Support Vector Machine) — Метод опорных векторов для регрессии
# Метод опорных векторов (SVM) в задаче регрессии пытается найти гиперплоскость, которая наилучшим образом описывает
# зависимость между признаками (время, цена открытия, лаговая цена) и целевой переменной (цена закрытия). В данном
# случае используется ядро RBF (радиальной базисной функции), которое позволяет выявить более сложные нелинейные
# зависимости.
#
# Как это работает в моем случае:
#
# SVM создает модель, которая находит зависимость между входными данными (время, цена открытия, цена закрытия с лагом)
# и целевой переменной (цена закрытия) с использованием ядра RBF, которое "изворачивает" данные в высокоразмерное
# пространство для поиска линейного решения в нем.
# Модель пытается минимизировать ошибку предсказания, используя лишь несколько ключевых "опорных" точек, которые
# оказывают наибольшее влияние на результаты.


# Lasso (линейная регрессия с L1-регуляризацией)
# Lasso (Least Absolute Shrinkage and Selection Operator) — это линейная регрессия с L1-регуляризацией.
# L1-регуляризация добавляет штраф за величину коэффициентов, что помогает избегать переобучения, и автоматически
# выполняет отбор признаков (в случае если коэффициент признака становится равен нулю, этот признак исключается).
#
# Как это работает в моем случае:
#
# Lasso создает линейную модель для прогнозирования цены закрытия на основе времени, цены открытия и цены закрытия с лагом.
# В отличие от обычной линейной регрессии, Lasso включает штраф за большие коэффициенты, что приводит к "сжатию"
# коэффициентов (некоторые могут стать нулевыми). Это помогает выявить наиболее важные признаки.
# Если признак незначителен, его коэффициент будет равен нулю, и модель будет игнорировать его влияние на прогноз.


# PLS (Partial Least Squares) — Метод частичных наименьших квадратов
# PLS — это метод регрессии, который пытается найти компоненты (или направления), которые максимизируют ковариацию
# между признаками и целевой переменной. В отличие от линейной регрессии, которая работает с исходными признаками
# напрямую, PLS находит новые компоненты, которые лучше всего объясняют как признаки, так и целевую переменную.
#
# Как это работает в моем случае:
#
# Модель PLS находит скрытые компоненты (направления), которые одновременно хорошо объясняют данные (время, цена
# открытия, цена закрытия с лагом) и целевую переменную (цена закрытия).
# Она проецирует исходные данные на эти компоненты и строит регрессионную модель в новом пространстве.
# Количество компонентов задается как гиперпараметр модели (в моем случае 3).
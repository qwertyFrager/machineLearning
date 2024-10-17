import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.cross_decomposition import PLSRegression

# Загрузка данных
file_path_2 = 'DataSet5_1.csv'  # Укажите правильный путь к файлу
data_prices = pd.read_csv(file_path_2, sep=',')

# Переименование столбцов для удобства
data_prices.columns = ['Дата мониторинга', 'Курица', 'Молоко', 'Мука', 'Хлеб', 'Яйцо', 'Картофель']

# Преобразуем первый столбец в формат datetime
data_prices['Дата мониторинга'] = pd.to_datetime(data_prices['Дата мониторинга'], format='%d.%m.%Y')

# Подготовим данные для анализа
data = pd.DataFrame()
data['x'] = (data_prices['Дата мониторинга'] - data_prices['Дата мониторинга'].min()).dt.days  # Время в днях
data['y'] = data_prices['Курица']  # Целевая переменная — цена на курицу
data['p1'] = data_prices['Молоко']
data['p2'] = data_prices['Мука']
data['p3'] = data_prices['Хлеб']
data['p4'] = data_prices['Яйцо']
data['p5'] = data_prices['Картофель']

# Определяем размер обучающей выборки (80% данных)
n = round(0.8 * len(data))

# Разделение на обучающую и тестовую выборки
train = data[:n]
test = data[n:]

# Визуализируем исходные данные
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train['x'], train['y'], color='green', linestyle='solid', label='Train Курица')
ax.plot(train['x'], train['p1'], color='blue', linestyle='solid', label='Train Молоко')
ax.plot(train['x'], train['p2'], color='red', linestyle='solid', label='Train Мука')
ax.plot(test['x'], test['y'], color='green', linestyle='dashed', label='Test Курица')
ax.plot(test['x'], test['p1'], color='blue', linestyle='dashed', label='Test Молоко')
ax.plot(test['x'], test['p2'], color='red', linestyle='dashed', label='Test Мука')
ax.set(title="Исходные данные: цены на продукты", xlabel="Время (в днях)", ylabel="Цена (руб.)")
ax.legend()
plt.show()

# Разделяем данные на признаки (x, p1, p2, p3, p4, p5) и целевую переменную (y)
train_x = train.drop(['y'], axis=1)
train_y = train['y']
test_x = test.drop(['y'], axis=1)
test_y = test['y']

# Прогнозирование методом МНК
def mapping_func(x, a, b, c, b1, b2, b3, b4, b5):
    return a + b * x['x'] + c * x['x']**2 + b1 * x['p1'] + b2 * x['p2'] + b3 * x['p3'] + b4 * x['p4'] + b5 * x['p5']

# Подбираем параметры модели МНК
args, _ = curve_fit(mapping_func, train_x, train_y)
a, b, c, b1, b2, b3, b4, b5 = args

# Прогнозируем значения для обучающей и тестовой выборки
res_y_tr = a + b * train['x'] + c * train['x']**2 + b1 * train['p1'] + b2 * train['p2'] + b3 * train['p3'] + b4 * train['p4'] + b5 * train['p5']
res_y_ts = a + b * test['x'] + c * test['x']**2 + b1 * test['p1'] + b2 * test['p2'] + b3 * test['p3'] + b4 * test['p4'] + b5 * test['p5']

# Визуализация результатов модели МНК
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train['x'], train['y'], color='green', linestyle='solid', label='Train Real')
ax.plot(test['x'], test['y'], color='green', linestyle='dashed', label='Test Real')
ax.plot(train['x'], res_y_tr, color='red', linestyle='solid', label='Train Predicted')
ax.plot(test['x'], res_y_ts, color='red', linestyle='dashed', label='Test Predicted')
ax.set(title="Прогнозирование цены курицы методом МНК", xlabel="Время (в днях)", ylabel="Цена (руб.)")
ax.legend()
plt.show()

# Прогнозирование методом SVM
svm_model = SVR(kernel='rbf')  # Используем радиальное базисное ядро
svm_model.fit(train_x, train_y)

# Прогнозируем значения для обучающей и тестовой выборки с использованием SVM
res_y_tr_svm = svm_model.predict(train_x)
res_y_ts_svm = svm_model.predict(test_x)

# Визуализация результатов модели SVM
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train['x'], train['y'], color='green', linestyle='solid', label='Train Real')
ax.plot(test['x'], test['y'], color='green', linestyle='dashed', label='Test Real')
ax.plot(train['x'], res_y_tr_svm, color='red', linestyle='solid', label='Train Predicted SVM')
ax.plot(test['x'], res_y_ts_svm, color='red', linestyle='dashed', label='Test Predicted SVM')
ax.set(title="Прогнозирование цены курицы методом SVM", xlabel="Время (в днях)", ylabel="Цена (руб.)")
ax.legend()
plt.show()

# Прогнозирование методом Lasso
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(train_x, train_y)

# Прогнозируем значения для обучающей и тестовой выборки с использованием Lasso
res_y_tr_lasso = lasso_model.predict(train_x)
res_y_ts_lasso = lasso_model.predict(test_x)

# Визуализация результатов модели Lasso
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train['x'], train['y'], color='green', linestyle='solid', label='Train Real')
ax.plot(test['x'], test['y'], color='green', linestyle='dashed', label='Test Real')
ax.plot(train['x'], res_y_tr_lasso, color='red', linestyle='solid', label='Train Predicted Lasso')
ax.plot(test['x'], res_y_ts_lasso, color='red', linestyle='dashed', label='Test Predicted Lasso')
ax.set(title="Прогнозирование цены курицы методом Lasso", xlabel="Время (в днях)", ylabel="Цена (руб.)")
ax.legend()
plt.show()

# Прогнозирование методом PLS
pls_model = PLSRegression(n_components=3)
pls_model.fit(train_x, train_y)

# Прогнозируем значения для обучающей и тестовой выборки с использованием PLS
res_y_tr_pls = pls_model.predict(train_x)
res_y_ts_pls = pls_model.predict(test_x)

# Визуализация результатов модели PLS
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train['x'], train['y'], color='green', linestyle='solid', label='Train Real')
ax.plot(test['x'], test['y'], color='green', linestyle='dashed', label='Test Real')
ax.plot(train['x'], res_y_tr_pls, color='red', linestyle='solid', label='Train Predicted PLS')
ax.plot(test['x'], res_y_ts_pls, color='red', linestyle='dashed', label='Test Predicted PLS')
ax.set(title="Прогнозирование цены курицы методом PLS", xlabel="Время (в днях)", ylabel="Цена (руб.)")
ax.legend()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.cross_decomposition import PLSRegression

# Загрузка данных
file_path_bikes = 'DataSet5_2.csv'  # Укажите правильный путь к файлу
data_bikes = pd.read_csv(file_path_bikes, sep=',')

# Подготовим данные: целевая переменная (count) и признаки (остальные параметры)
data_bikes_prepared = pd.DataFrame()
data_bikes_prepared['y'] = data_bikes['count']  # Целевая переменная - количество арендованных велосипедов
data_bikes_prepared['p1'] = data_bikes['holiday']  # Праздничный день
data_bikes_prepared['p2'] = data_bikes['humidity']  # Влажность
data_bikes_prepared['p3'] = data_bikes['registered']  # Количество зарегистрированных пользователей
data_bikes_prepared['p4'] = data_bikes['summer']  # Лето
data_bikes_prepared['p5'] = data_bikes['temp']  # Температура
data_bikes_prepared['p6'] = data_bikes['windspeed']  # Скорость ветра
data_bikes_prepared['p7'] = data_bikes['workingday']  # Рабочий день

# Определяем размер обучающей выборки (80% данных)
n = round(0.8 * len(data_bikes_prepared))

# Разделим данные на обучающую и тестовую выборки
train = data_bikes_prepared[:n]
test = data_bikes_prepared[n:]

# Визуализация данных
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train.index, train['y'], color='green', linestyle='solid', label='Train Count')
ax.plot(train.index, train['p2'], color='blue', linestyle='solid', label='Train Humidity')
ax.plot(test.index, test['y'], color='green', linestyle='dashed', label='Test Count')
ax.plot(test.index, test['p2'], color='blue', linestyle='dashed', label='Test Humidity')
ax.set(title="Исходные данные: аренда велосипедов", xlabel="Индекс", ylabel="Значения")
ax.legend()
plt.show()

# Разделение данных на признаки и целевую переменную
train_x = train.drop(['y'], axis=1)
train_y = train['y']
test_x = test.drop(['y'], axis=1)
test_y = test['y']

# Прогнозирование методом МНК
def mapping_func_bikes(x, a, b1, b2, b3, b4, b5, b6, b7):
    return a + b1 * x['p1'] + b2 * x['p2'] + b3 * x['p3'] + b4 * x['p4'] + b5 * x['p5'] + b6 * x['p6'] + b7 * x['p7']

args, _ = curve_fit(mapping_func_bikes, train_x, train_y)
a, b1, b2, b3, b4, b5, b6, b7 = args

res_y_tr = a + b1 * train_x['p1'] + b2 * train_x['p2'] + b3 * train_x['p3'] + b4 * train_x['p4'] + b5 * train_x['p5'] + b6 * train_x['p6'] + b7 * train_x['p7']
res_y_ts = a + b1 * test_x['p1'] + b2 * test_x['p2'] + b3 * test_x['p3'] + b4 * test_x['p4'] + b5 * test_x['p5'] + b6 * test_x['p6'] + b7 * test_x['p7']

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train.index, train['y'], color='green', linestyle='solid', label='Train Real')
ax.plot(test.index, test['y'], color='green', linestyle='dashed', label='Test Real')
ax.plot(train.index, res_y_tr, color='red', linestyle='solid', label='Train Predicted')
ax.plot(test.index, res_y_ts, color='red', linestyle='dashed', label='Test Predicted')
ax.set(title="Прогнозирование количества арендованных велосипедов методом МНК", xlabel="Индекс", ylabel="Количество")
ax.legend()
plt.show()

# Прогнозирование методом SVM
svm_model = SVR(kernel='rbf')
svm_model.fit(train_x, train_y)

res_y_tr_svm = svm_model.predict(train_x)
res_y_ts_svm = svm_model.predict(test_x)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train.index, train['y'], color='green', linestyle='solid', label='Train Real')
ax.plot(test.index, test['y'], color='green', linestyle='dashed', label='Test Real')
ax.plot(train.index, res_y_tr_svm, color='red', linestyle='solid', label='Train Predicted SVM')
ax.plot(test.index, res_y_ts_svm, color='red', linestyle='dashed', label='Test Predicted SVM')
ax.set(title="Прогнозирование количества арендованных велосипедов методом SVM", xlabel="Индекс", ylabel="Количество")
ax.legend()
plt.show()

# Прогнозирование методом Lasso
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(train_x, train_y)

res_y_tr_lasso = lasso_model.predict(train_x)
res_y_ts_lasso = lasso_model.predict(test_x)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train.index, train['y'], color='green', linestyle='solid', label='Train Real')
ax.plot(test.index, test['y'], color='green', linestyle='dashed', label='Test Real')
ax.plot(train.index, res_y_tr_lasso, color='red', linestyle='solid', label='Train Predicted Lasso')
ax.plot(test.index, res_y_ts_lasso, color='red', linestyle='dashed', label='Test Predicted Lasso')
ax.set(title="Прогнозирование количества арендованных велосипедов методом Lasso", xlabel="Индекс", ylabel="Количество")
ax.legend()
plt.show()

# Прогнозирование методом PLS
pls_model = PLSRegression(n_components=3)
pls_model.fit(train_x, train_y)

res_y_tr_pls = pls_model.predict(train_x)
res_y_ts_pls = pls_model.predict(test_x)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train.index, train['y'], color='green', linestyle='solid', label='Train Real')
ax.plot(test.index, test['y'], color='green', linestyle='dashed', label='Test Real')
ax.plot(train.index, res_y_tr_pls, color='red', linestyle='solid', label='Train Predicted PLS')
ax.plot(test.index, res_y_ts_pls, color='red', linestyle='dashed', label='Test Predicted PLS')
ax.set(title="Прогнозирование количества арендованных велосипедов методом PLS", xlabel="Индекс", ylabel="Количество")
ax.legend()
plt.show()

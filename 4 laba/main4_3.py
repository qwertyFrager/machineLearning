import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from sklearn.neighbors import KNeighborsRegressor
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error

# Шаг 1: Загрузка и подготовка данных
data_oil_dollar = pd.read_csv('DataSet3_2.csv', sep=';')

# Преобразование данных
data_oil_dollar['Data'] = pd.to_datetime(data_oil_dollar['Data'], format='%d.%m.%Y')
data_oil_dollar['Dollar'] = data_oil_dollar['Dollar'].str.replace(',', '.').astype(float)
data_oil_dollar['Oil Brent'] = data_oil_dollar['Oil Brent'].str.replace(',', '.').astype(float)

# Определим диапазон для выборки
start_idx = 0  # С самого начала
end_idx = len(data_oil_dollar)  # До конца данных

# Извлечение временной шкалы (даты), курса доллара и стоимости нефти
x = data_oil_dollar['Data'][start_idx:end_idx].to_numpy()
dollar = data_oil_dollar['Dollar'][start_idx:end_idx].to_numpy()
oil_brent = data_oil_dollar['Oil Brent'][start_idx:end_idx].to_numpy()

# Разделение на тренировочные и тестовые выборки (80%/20%)
n = round(0.8 * len(oil_brent))

train_x = x[:n]
train_dollar = dollar[:n]
train_oil_brent = oil_brent[:n]

test_x = x[n:]
test_dollar = dollar[n:]
test_oil_brent = oil_brent[n:]

# Преобразование дат в числовые значения (дни от начальной даты)
train_x_numeric = (train_x - train_x[0]).astype('timedelta64[D]').astype(int)
test_x_numeric = (test_x - train_x[0]).astype('timedelta64[D]').astype(int)

# Шаг 2: Полиномиальная регрессия (МНК)

# Определение полинома 2-й степени для регрессии
def mapping_func(x, c1, c2, c3):
    return c1 * x**2 + c2 * x + c3

# Подбор параметров модели МНК на тренировочных данных
args, covar = curve_fit(mapping_func, train_x_numeric, train_oil_brent)
c1, c2, c3 = args

# Прогноз на обучающей выборке
res_y_train_mnk = c1 * train_x_numeric**2 + c2 * train_x_numeric + c3

# Визуализация результатов МНК на обучающих данных
fig = go.Figure()
fig.add_trace(go.Scatter(x=train_x_numeric, y=train_oil_brent, line=dict(color='green', width=2), name='Тренировочные данные'))
fig.add_trace(go.Scatter(x=train_x_numeric, y=res_y_train_mnk, line=dict(color='red', width=2), name='Прогнозы МНК'))
fig.update_layout(title="Полиномиальная регрессия (МНК) на тренировочных данных", xaxis_title="Время (дни)", yaxis_title="Стоимость нефти (Brent)")
fig.show()

# Прогноз на тестовой выборке
res_y_test_mnk = c1 * test_x_numeric**2 + c2 * test_x_numeric + c3

# Визуализация результатов МНК на тестовых данных
fig = go.Figure()
fig.add_trace(go.Scatter(x=test_x_numeric, y=test_oil_brent, line=dict(color='green', width=2), name='Тестовые данные'))
fig.add_trace(go.Scatter(x=test_x_numeric, y=res_y_test_mnk, line=dict(color='red', width=2), name='Прогнозы МНК'))
fig.update_layout(title="Полиномиальная регрессия (МНК) на тестовых данных", xaxis_title="Время (дни)", yaxis_title="Стоимость нефти (Brent)")
fig.show()

# Шаг 3: Модель kNN

# Построение модели kNN для прогнозирования стоимости нефти
neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(train_x_numeric.reshape(-1, 1), train_oil_brent)

# Прогноз на обучающей выборке
res_y_train_knn = neigh.predict(train_x_numeric.reshape(-1, 1))

# Визуализация результатов kNN на обучающих данных
fig = go.Figure()
fig.add_trace(go.Scatter(x=train_x_numeric, y=train_oil_brent, line=dict(color='green', width=2), name='Тренировочные данные'))
fig.add_trace(go.Scatter(x=train_x_numeric, y=res_y_train_knn, line=dict(color='red', width=2), name='Прогнозы kNN'))
fig.update_layout(title="kNN на тренировочных данных", xaxis_title="Время (дни)", yaxis_title="Стоимость нефти (Brent)")
fig.show()

# Прогноз на тестовой выборке
res_y_test_knn = neigh.predict(test_x_numeric.reshape(-1, 1))

# Визуализация результатов kNN на тестовых данных
fig = go.Figure()
fig.add_trace(go.Scatter(x=test_x_numeric, y=test_oil_brent, line=dict(color='green', width=2), name='Тестовые данные'))
fig.add_trace(go.Scatter(x=test_x_numeric, y=res_y_test_knn, line=dict(color='red', width=2), name='Прогнозы kNN'))
fig.update_layout(title="kNN на тестовых данных", xaxis_title="Время (дни)", yaxis_title="Стоимость нефти (Brent)")
fig.show()

# Шаг 4: Модель ARIMA

# Прогнозирование с помощью ARIMA для стоимости нефти
history = [y for y in train_oil_brent]
predictions = []

# Прогнозируем шаг за шагом на тестовой выборке
for t in range(len(test_oil_brent)):
    model = sm.tsa.arima.ARIMA(history, order=(1, 1, 0))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
    history.append(test_oil_brent[t])

# Визуализация результатов ARIMA на тестовых данных
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.concatenate((train_x_numeric, test_x_numeric)), y=np.concatenate((train_oil_brent, test_oil_brent)), line=dict(color='green', width=2), name='Реальные данные'))
fig.add_trace(go.Scatter(x=test_x_numeric, y=predictions, line=dict(color='red', width=2), name='Прогнозы ARIMA'))
fig.update_layout(title="ARIMA на тестовых данных", xaxis_title="Время (дни)", yaxis_title="Стоимость нефти (Brent)")
fig.show()

# Шаг 5: Оценка точности моделей (MAE)

# Средняя абсолютная ошибка (MAE) для каждой модели
mae_mnk = mean_absolute_error(test_oil_brent, res_y_test_mnk)
mae_knn = mean_absolute_error(test_oil_brent, res_y_test_knn)
mae_arima = mean_absolute_error(test_oil_brent, predictions)

# Вывод точности моделей
print(f"Средняя абсолютная ошибка (MAE) для МНК: {mae_mnk:.4f}")
print(f"Средняя абсолютная ошибка (MAE) для kNN: {mae_knn:.4f}")
print(f"Средняя абсолютная ошибка (MAE) для ARIMA: {mae_arima:.4f}")

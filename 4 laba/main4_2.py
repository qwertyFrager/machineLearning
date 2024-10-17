import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from sklearn.neighbors import KNeighborsRegressor
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error

# Шаг 1: Загрузка данных
# Мы загружаем данные из CSV файла и преобразуем временные метки в формат datetime.
data_energy = pd.read_csv('DataSet3_1.csv')
data_energy['Timestamp'] = pd.to_datetime(data_energy['Timestamp'])

# Шаг 2: Подготовка данных
# Выбираем данные между индексами 100 и 300, чтобы построить модели и протестировать их.
start_idx = 100
end_idx = 300

x = data_energy['Timestamp'][start_idx:end_idx].to_numpy()
y = data_energy['Power (kW)'][start_idx:end_idx].to_numpy()

# Разделение на обучающую и тестовую выборки (80%/20%)
n = round(0.8 * (end_idx - start_idx))
train_x = x[:n]
train_y = y[:n]
test_x = x[n:]
test_y = y[n:]

# Преобразование времени в количество минут от начальной точки для моделей МНК и kNN
train_x_numeric = (train_x - train_x[0]).astype('timedelta64[m]').astype(int)  # минуты от начальной точки
test_x_numeric = (test_x - train_x[0]).astype('timedelta64[m]').astype(int)

# Шаг 3: Модель наименьших квадратов (МНК)
# Полиномиальная регрессия второго порядка для нахождения общей тенденции в данных
def mapping_func(x, c1, c2, c3):
    return c1 * x**2 + c2 * x + c3

# Подбор параметров для полинома 2-й степени
args, covar = curve_fit(mapping_func, train_x_numeric, train_y)
c1, c2, c3 = args

# Прогноз на обучающей выборке
res_y_train_mnk = c1 * train_x_numeric**2 + c2 * train_x_numeric + c3

# Визуализация результатов МНК на обучающих данных
fig = go.Figure()
fig.add_trace(go.Scatter(x=train_x_numeric, y=train_y, line=dict(color='green', width=2), name='Тренировочные данные'))
fig.add_trace(go.Scatter(x=train_x_numeric, y=res_y_train_mnk, line=dict(color='red', width=2), name='Прогнозы МНК'))
fig.update_layout(title="Полиномиальная регрессия (МНК) на тренировочных данных",
                  xaxis_title="Время (минуты от начала выборки)", yaxis_title="Потребляемая мощность (kW)",
                  template="plotly_white")
fig.show()

# Прогноз на тестовой выборке
res_y_test_mnk = c1 * test_x_numeric**2 + c2 * test_x_numeric + c3

# Визуализация результатов МНК на тестовых данных
fig = go.Figure()
fig.add_trace(go.Scatter(x=test_x_numeric, y=test_y, line=dict(color='green', width=2), name='Тестовые данные'))
fig.add_trace(go.Scatter(x=test_x_numeric, y=res_y_test_mnk, line=dict(color='red', width=2), name='Прогнозы МНК'))
fig.update_layout(title="Полиномиальная регрессия (МНК) на тестовых данных",
                  xaxis_title="Время (минуты от начала выборки)", yaxis_title="Потребляемая мощность (kW)",
                  template="plotly_white")
fig.show()

# Шаг 4: Модель ближайших соседей (kNN)
# Модель ближайших соседей на обучающих данных
neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(train_x_numeric.reshape(-1, 1), train_y)

# Прогноз на обучающей выборке
res_y_train_knn = neigh.predict(train_x_numeric.reshape(-1, 1))

# Визуализация результатов kNN на обучающих данных
fig = go.Figure()
fig.add_trace(go.Scatter(x=train_x_numeric, y=train_y, line=dict(color='green', width=2), name='Тренировочные данные'))
fig.add_trace(go.Scatter(x=train_x_numeric, y=res_y_train_knn, line=dict(color='red', width=2), name='Прогнозы kNN'))
fig.update_layout(title="kNN на тренировочных данных",
                  xaxis_title="Время (минуты от начала выборки)", yaxis_title="Потребляемая мощность (kW)",
                  template="plotly_white")
fig.show()

# Прогноз на тестовой выборке
res_y_test_knn = neigh.predict(test_x_numeric.reshape(-1, 1))

# Визуализация результатов kNN на тестовых данных
fig = go.Figure()
fig.add_trace(go.Scatter(x=test_x_numeric, y=test_y, line=dict(color='green', width=2), name='Тестовые данные'))
fig.add_trace(go.Scatter(x=test_x_numeric, y=res_y_test_knn, line=dict(color='red', width=2), name='Прогнозы kNN'))
fig.update_layout(title="kNN на тестовых данных",
                  xaxis_title="Время (минуты от начала выборки)", yaxis_title="Потребляемая мощность (kW)",
                  template="plotly_white")
fig.show()

# Шаг 5: Модель ARIMA
# Прогнозирование на основе временных данных с помощью модели ARIMA
history = [y for y in train_y]
predictions = []

# Прогнозируем шаг за шагом на тестовой выборке
for t in range(len(test_y)):
    model = sm.tsa.arima.ARIMA(history, order=(1, 1, 0))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
    history.append(test_y[t])

# Визуализация результатов ARIMA на тестовых данных
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.concatenate((train_x_numeric, test_x_numeric)), y=np.concatenate((train_y, test_y)),
                         line=dict(color='green', width=2), name='Реальные данные'))
fig.add_trace(go.Scatter(x=test_x_numeric, y=predictions, line=dict(color='red', width=2), name='Прогнозы ARIMA'))
fig.update_layout(title="ARIMA на тестовых данных",
                  xaxis_title="Время (минуты от начала выборки)", yaxis_title="Потребляемая мощность (kW)",
                  template="plotly_white")
fig.show()

# Шаг 6: Оценка точности моделей (MAE)
# Средняя абсолютная ошибка (MAE) позволяет оценить, насколько точны наши прогнозы
mae_mnk = mean_absolute_error(test_y, res_y_test_mnk)
mae_knn = mean_absolute_error(test_y, res_y_test_knn)
mae_arima = mean_absolute_error(test_y, predictions)

# Вывод точности моделей
print(f"Средняя абсолютная ошибка (MAE) для МНК: {mae_mnk:.4f}")
print(f"Средняя абсолютная ошибка (MAE) для kNN: {mae_knn:.4f}")
print(f"Средняя абсолютная ошибка (MAE) для ARIMA: {mae_arima:.4f}")

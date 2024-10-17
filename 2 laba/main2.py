# Импортируем необходимые библиотеки
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

# Читаем файл данных
file_path = 'DataSet1_2.csv'  # Укажите ваш путь к файлу
data = pd.read_csv(file_path, delimiter='|')

# Выводим исходный датасет
print("Оригинальные данные:\n", data.head())

# 1. Проверка модели на НЕОБРАБОТАННЫХ данных (без фильтрации дисперсии и корреляции)

# Определяем целевую переменную (fraud) и признаки
X_raw = data.drop(columns=['fraud'])  # Все признаки кроме 'fraud'
y_raw = data['fraud']  # Целевая переменная

# Разделение на обучающую и тестовую выборки
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

# Обучаем модель на необработанных данных
model_raw = RandomForestClassifier(random_state=42)
model_raw.fit(X_train_raw, y_train_raw)

# Предсказываем результаты на тестовой выборке и выводим отчет
y_pred_raw = model_raw.predict(X_test_raw)
classification_rep_raw = classification_report(y_test_raw, y_pred_raw)
accuracy_raw = accuracy_score(y_test_raw, y_pred_raw)

print("Результаты классификации на необработанных данных:\n", classification_rep_raw)
print(f"Точность модели на необработанных данных: {accuracy_raw * 100:.2f}%\n")


# 2. Добавляем новые вычисляемые показатели

# Вычисляем дополнительные показатели, которые могут помочь улучшить предсказание
data['ne_otm'] = data['totalScanTimeInSeconds'] * data['scannedLineItemsPerSecond']
data['otm_i_ne_otm'] = data['lineItemVoids'] + data['ne_otm']
data['sec_na_1_udach_scan'] = data['totalScanTimeInSeconds'] / data['otm_i_ne_otm']
data['udach_i_neudach_scan'] = data['otm_i_ne_otm'] + data['scansWithoutRegistration']
data['dolya_neudach_scan'] = data['scansWithoutRegistration'] / data['udach_i_neudach_scan']
data['sec_na_1_scan'] = data['totalScanTimeInSeconds'] / data['udach_i_neudach_scan']

# Выводим датасет с новыми показателями
print("Данные с новыми показателями:\n", data.head())


# 3. Удаление колонок с низкой дисперсией

# Признаки с низкой дисперсией не добавляют информации для разделения классов, так как их значения почти одинаковы для всех наблюдений
numerical_data = data.select_dtypes(include=['float64', 'int64'])
selector = VarianceThreshold(threshold=0.01)  # Удаляем признаки с дисперсией меньше 0.01
filtered_data = selector.fit_transform(numerical_data)
remaining_columns = numerical_data.columns[selector.get_support()]
filtered_df = pd.DataFrame(filtered_data, columns=remaining_columns)

# Выводим отфильтрованные данные без колонок с низкой дисперсией
print("Отфильтрованные данные без колонок с низкой дисперсией:\n", filtered_df.head())


# 4. Построение матрицы корреляции и удаление признаков с высокой корреляцией (зачем это нужно?)

# Матрица корреляции показывает взаимосвязи между признаками. Признаки с высокой корреляцией (>0.75) могут дублировать информацию
correlation_matrix = filtered_df.corr()

# Визуализируем матрицу корреляции с помощью seaborn
plt.figure(figsize=(10, 8))
sb.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Матрица корреляции")
plt.show()

# Удаляем признаки с высокой корреляцией (больше 0.75), чтобы не было дублирования информации
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
to_drop_high_corr = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.75)]
final_df = filtered_df.drop(columns=to_drop_high_corr)

# Выводим финальный набор данных без коррелированных признаков
print("Финальный набор данных без коррелированных признаков:\n", final_df.head())


# 5. Обучение модели на обработанных данных

# Определяем признаки и целевую переменную
X_processed = final_df.drop(columns=['fraud'])
y_processed = final_df['fraud']

# Разделяем на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)

# Обучаем модель на обработанных данных
model_processed = RandomForestClassifier(random_state=42)
model_processed.fit(X_train, y_train)

# Предсказываем результаты на тестовой выборке
y_pred_processed = model_processed.predict(X_test)

# Выводим отчет классификации
classification_rep_processed = classification_report(y_test, y_pred_processed)
accuracy_processed = accuracy_score(y_test, y_pred_processed)

print("Результаты классификации на обработанных данных:\n", classification_rep_processed)
print(f"Точность модели на обработанных данных: {accuracy_processed * 100:.2f}%\n")


# 6. Сравнение результатов
print(f"Сравнение точности модели:\n"
      f"Точность на необработанных данных: {accuracy_raw * 100:.2f}%\n"
      f"Точность на обработанных данных: {accuracy_processed * 100:.2f}%\n")

# Описание результата:
# Точность (accuracy) — это доля правильно предсказанных наблюдений к общему числу наблюдений
# В данном случае точность модели составляет 98%, что означает, что 98% всех наблюдений
# были правильно классифицированы как мошеннические или немошеннические

# Была использована модель RandomForestClassifier (Случайный лес)

# Каждый отдельный признак может не сильно коррелировать с целевым значением, но их комбинация может давать значимые
# результаты. Случайный лес хорошо выявляет такие комбинации признаков, которые в совокупности способны лучше
# разделять классы.

# Корреляция показывает только линейную зависимость, но связи между признаками и целевым значением могут быть
# нелинейными. Модель случайного леса эффективно решает такие задачи, выявляя сложные нелинейные зависимости между
# признаками.

# В каждом отдельном дереве решений модель находит свои комбинации признаков, которые дают наилучшее разделение
# классов, и даже если эти деревья обучены на различных случайных подвыборках данных, их совокупные результаты могут
# дать хорошее предсказание.

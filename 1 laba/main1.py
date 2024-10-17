import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Первый датасет
# Загрузка данных
df = pd.read_csv('DataSet1_1.csv', sep=';')

# Убираем столбец "N" и оставляем только данные по проектам
df_projects = df.drop(columns=['N'])

# Извлекаем только числовые данные (показатели X1-X20)
data = df_projects.iloc[:, 1:21].values  # Без столбца "Region"

# Нормализация данных (стандартизация)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Применение метода K-средних для кластеризации (возьмем, например, 4 кластера)
kmeans = KMeans(n_clusters=4, random_state=42)
df_projects['Cluster'] = kmeans.fit_predict(data_scaled)

# Визуализация кластеров с использованием PCA (метод главных компонент) для снижения размерности до 2D
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=df_projects['Cluster'], palette='Set2', s=100)
plt.title('Кластеры регионов на основе показателей ИТ-проектов (2004-2024)')
plt.xlabel('Первая главная компонента')
plt.ylabel('Вторая главная компонента')
plt.legend(title='Кластер')
plt.show()

# Вывод кластеров для каждого региона
print(df_projects[['Region', 'Cluster']])


# Регионы в одном кластере могут иметь:
# Схожую динамику роста: Например, все регионы в кластере могут демонстрировать устойчивый рост числа IT-проектов, хотя конкретные значения проектов в каждом году могут различаться.
# Схожие тренды в определенные годы: Например, регионы могли иметь резкий рост проектов в одни и те же годы (например, скачок в 2020 году), а затем стабильность.
# Стабильное количество проектов: Некоторые регионы могут быть сгруппированы, если число IT-проектов у них было стабильно на протяжении всех лет.

# PCA — это метод линейного преобразования данных, который находит направления (компоненты), вдоль которых данные варьируются максимально. Эти направления называются главными компонентами.
# Как это работает:
# 1) Поиск главных осей: PCA анализирует корреляцию между признаками и находит новые оси (направления) в пространстве данных, вдоль которых разброс данных максимален. Эти оси и называются главными компонентами.
# 2) Сортировка компонент: Первая главная компонента (PC1) — это направление, вдоль которого вариация (разброс) данных максимальна. Вторая главная компонента (PC2) — это следующее направление, которое перпендикулярно первой компоненте и описывает вторую по величине вариацию данных. И так далее для последующих компонент.
# 3) Уменьшение размерности: В реальных данных часто много признаков, и некоторые из них могут быть избыточны или сильно коррелированы. PCA позволяет преобразовать данные в новое пространство с меньшим числом осей (компонент), при этом сохраняя основную информацию о вариации в данных. Например, если у нас 20 признаков (как в твоем случае), мы можем уменьшить их до 2-3 главных компонент, сохранив при этом большую часть информации.


# Второй датасет
# Загрузка данных
df_fraud = pd.read_csv('DataSet1_2.csv', sep='|')

# Установим размер графиков
plt.figure(figsize=(14, 10))

# Построим гистограммы для всех числовых признаков
df_fraud.hist(bins=20, figsize=(14, 12), layout=(4, 3), edgecolor='black')
plt.suptitle('Распределение признаков в данных о кражах', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Построим корреляционную матрицу
plt.figure(figsize=(12, 10))
corr_matrix = df_fraud.corr()

# Построим тепловую карту корреляций
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Корреляционная матрица признаков', fontsize=16)
plt.show()

# Гистограммы помогут проанализировать распределение каждого признака, что даст визуальное представление о том, как варьируются данные.
# Корреляционная матрица покажет взаимосвязи между признаками, что поможет выделить ключевые переменные для классификации и оптимизации модели.


# Третий датасет
# Загрузка данных
df_stocks = pd.read_csv('DataSet1_3.csv')

# Преобразуем столбец 'Date' в формат даты
df_stocks['Date'] = pd.to_datetime(df_stocks['Date'])

# Построим график изменения цены закрытия акций Google
plt.figure(figsize=(12, 6))
plt.plot(df_stocks['Date'], df_stocks['Close'], label='Цена закрытия')
plt.title('Изменение цены закрытия акций Google')
plt.xlabel('Дата')
plt.ylabel('Цена закрытия ($)')
plt.grid(True)
plt.legend()
plt.show()

# Построим график изменения объема торгов
plt.figure(figsize=(12, 6))
plt.plot(df_stocks['Date'], df_stocks['Volume'], label='Объем торгов', color='orange')
plt.title('Изменение объема торгов акций Google')
plt.xlabel('Дата')
plt.ylabel('Объем торгов')
plt.grid(True)
plt.legend()
plt.show()

# Создадим категории для объема торгов
df_stocks['Volume_Category'] = pd.cut(df_stocks['Volume'], bins=5, labels=["Очень низкий", "Низкий", "Средний", "Высокий", "Очень высокий"])

# Посчитаем количество для каждой категории
volume_distribution = df_stocks['Volume_Category'].value_counts()

# Построим круговую диаграмму для распределения объема торгов
plt.figure(figsize=(8, 8))
plt.pie(volume_distribution, labels=volume_distribution.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0'])
plt.title('Распределение объема торгов акциями Google по категориям')
plt.show()

# Скрипичная диаграмма для анализа распределения цены закрытия
plt.figure(figsize=(12, 6))
sns.violinplot(x='Name', y='Close', data=df_stocks, inner='quartile')
plt.title('Скрипичная диаграмма для цены закрытия акций Google')
plt.xlabel('Компания')
plt.ylabel('Цена закрытия ($)')
plt.grid(True)
plt.show()


# Четвертый датасет
# Загрузка данных
df_covid = pd.read_csv('DataSet1_4.csv')

# Преобразуем столбец даты в формат datetime
df_covid['Day'] = pd.to_datetime(df_covid['Day'])

# Группируем данные по датам и суммируем новые и накопленные случаи
df_global = df_covid.groupby('Day').agg({
    'Daily new confirmed cases due to COVID-19 (rolling 7-day average, right-aligned)': 'sum',
    'Total confirmed cases of COVID-19': 'sum'
}).reset_index()

# Построим график суммарных новых случаев COVID-19 по всем странам
plt.figure(figsize=(10, 6))
plt.plot(df_global['Day'], df_global['Daily new confirmed cases due to COVID-19 (rolling 7-day average, right-aligned)'])
plt.title('Суммарные новые случаи COVID-19 по всем странам')
plt.xlabel('Дата')
plt.ylabel('Новые случаи (7-дневное среднее)')
plt.grid(True)
plt.show()

# Построим график суммарных накопленных случаев COVID-19 по всем странам
plt.figure(figsize=(10, 6))
plt.plot(df_global['Day'], df_global['Total confirmed cases of COVID-19'])
plt.title('Суммарные накопленные случаи COVID-19 по всем странам')
plt.xlabel('Дата')
plt.ylabel('Накопленные случаи')
plt.grid(True)
plt.show()
# Импортируем необходимые библиотеки
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
from dtaidistance import dtw
from statsmodels.tsa.seasonal import seasonal_decompose

# Путь к файлам данных. Замените пути на свои, если файлы находятся в другом месте.
files = {
    'normative': ['h30hz10.txt', 'h30hz20.txt', 'h30hz40.txt'],
    'non_normative': ['b30hz10.txt', 'b30hz20.txt', 'b30hz40.txt']
}

# 1. Загрузка данных
# Словарь для хранения загруженных данных
data = {'normative': [], 'non_normative': []}

# Загружаем данные из файлов и выбираем первые 100 значений
for category, file_list in files.items():
    for file_path in file_list:
        df = pd.read_csv(file_path, sep='\s+', header=None).iloc[:100]
        data[category].append(df)


# 2. Расчет коэффициентов корреляции
def calculate_correlations(df):
    """
    Функция для расчета коэффициентов корреляции (Пирсона, Спирмена, Кендала) между всеми парами столбцов в DataFrame.
    """
    corr_results = {}
    for i in range(df.shape[1]):
        for j in range(i + 1, df.shape[1]):
            pearson_corr, _ = pearsonr(df[i], df[j])
            spearman_corr, _ = spearmanr(df[i], df[j])
            kendall_corr, _ = kendalltau(df[i], df[j])
            corr_results[(i, j)] = {
                'pearson': pearson_corr,
                'spearman': spearman_corr,
                'kendall': kendall_corr
            }
    return corr_results


# Рассчитываем корреляции для каждой категории файлов
correlations = {'normative': [], 'non_normative': []}
for category, dfs in data.items():
    correlations[category] = [calculate_correlations(df) for df in dfs]


# 3. Расчет расстояний DTW (динамического времени)
def calculate_dtw_distances(df):
    """
    Функция для вычисления расстояний DTW между всеми парами столбцов в DataFrame.
    """
    dtw_results = {}
    for i in range(df.shape[1]):
        for j in range(i + 1, df.shape[1]):
            distance = dtw.distance(df[i].values, df[j].values)
            dtw_results[(i, j)] = distance
    return dtw_results


# Рассчитываем DTW расстояния для всех файлов
dtw_distances = {'normative': [], 'non_normative': []}
for category, dfs in data.items():
    dtw_distances[category] = [calculate_dtw_distances(df) for df in dfs]


# 4. Спектральный анализ
def plot_spectral_density(df, file_name):
    """
    Функция для вычисления и отображения спектральной плотности для каждого датчика в файле.
    """
    plt.figure(figsize=(12, 8))
    for i in range(df.shape[1]):
        freqs = np.fft.fftfreq(len(df[i]))
        fft_values = np.fft.fft(df[i].values)
        power_spectrum = np.abs(fft_values) ** 2
        plt.plot(freqs[:len(freqs) // 2], power_spectrum[:len(power_spectrum) // 2], label=f'Датчик {i + 1}')

    plt.title(f'Спектральная плотность - {file_name}')
    plt.xlabel('Частота')
    plt.ylabel('Мощность')
    plt.legend()
    plt.grid(True)
    plt.show()


# Отображаем спектральную плотность для каждого файла
for category, dfs in data.items():
    for idx, df in enumerate(dfs):
        file_name = files[category][idx]
        plot_spectral_density(df, file_name)


# 5. Разложение временных рядов
def decompose_and_plot(df, file_name):
    """
    Функция для разложения временного ряда на тренд, сезонные и шумовые компоненты с отображением результатов.
    """
    plt.figure(figsize=(15, 10))
    for i in range(df.shape[1]):
        decomposition = seasonal_decompose(df[i], model='additive', period=10)

        plt.subplot(df.shape[1], 4, i * 4 + 1)
        plt.plot(decomposition.observed)
        plt.title(f'Датчик {i + 1} - Наблюдаемые данные')

        plt.subplot(df.shape[1], 4, i * 4 + 2)
        plt.plot(decomposition.trend)
        plt.title(f'Датчик {i + 1} - Тренд')

        plt.subplot(df.shape[1], 4, i * 4 + 3)
        plt.plot(decomposition.seasonal)
        plt.title(f'Датчик {i + 1} - Сезонная компонента')

        plt.subplot(df.shape[1], 4, i * 4 + 4)
        plt.plot(decomposition.resid)
        plt.title(f'Датчик {i + 1} - Шум')

    plt.suptitle(f'Разложение временного ряда - {file_name}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Выполняем разложение для всех файлов
for category, dfs in data.items():
    for idx, df in enumerate(dfs):
        file_name = files[category][idx]
        decompose_and_plot(df, file_name)

# Вывод результатов корреляций и DTW расстояний
print("Корреляции и DTW расстояния для каждого файла:")

for category in correlations.keys():
    print(f"\n{category.upper()}")
    for idx, (corr_result, dtw_result) in enumerate(zip(correlations[category], dtw_distances[category])):
        file_name = files[category][idx]
        print(f"\nФайл: {file_name}")

        print("Корреляции:")
        for (i, j), corr_vals in corr_result.items():
            print(f"  Датчики {i + 1}-{j + 1}: Пирсон={corr_vals['pearson']:.3f}, "
                  f"Спирмен={corr_vals['spearman']:.3f}, Кендалл={corr_vals['kendall']:.3f}")

        print("DTW расстояния:")
        for (i, j), distance in dtw_result.items():
            print(f"  Датчики {i + 1}-{j + 1}: DTW расстояние = {distance:.3f}")

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from fastdtw import fastdtw

# Определяем свою функцию для евклидова расстояния
def euclidean(a, b):
    return abs(a - b)

# Загрузка данных из файлов
files = {
    'b30hz10': 'b30hz10.txt',
    'b30hz20': 'b30hz20.txt',
    'b30hz40': 'b30hz40.txt',
    'h30hz20': 'h30hz20.txt',
    'h30hz30': 'h30hz30.txt',
    'h30hz50': 'h30hz50.txt'
}

# Чтение файлов и извлечение первых 100 значений по каждому каналу (4 канала на файл)
data = {name: pd.read_csv(file, sep='\t', header=None).iloc[:100] for name, file in files.items()}


# Функция для расчета коэффициентов корреляции
def calculate_correlations(data1, data2):
    pearson_corr = [pearsonr(data1.iloc[:, i], data2.iloc[:, i])[0] for i in range(4)]
    spearman_corr = [spearmanr(data1.iloc[:, i], data2.iloc[:, i])[0] for i in range(4)]
    kendall_corr = [kendalltau(data1.iloc[:, i], data2.iloc[:, i])[0] for i in range(4)]

    return pearson_corr, spearman_corr, kendall_corr

# Функция рассчитывает коэффициенты корреляции для всех четырех каналов вибрации:
# Корреляция Пирсона: показывает линейную зависимость между сигналами
# Корреляция Спирмена: используется для нелинейных зависимостей (ранговая корреляция)
# Корреляция Кендалла: также ранговая корреляция, но с другим способом подсчета

# Функция возвращает три списка (по одному для каждого типа корреляции)


# Функция для расчета DTW
def calculate_dtw(data1, data2):
    dtw_distances = [fastdtw(data1.iloc[:, i].to_numpy(), data2.iloc[:, i].to_numpy(), dist=euclidean)[0] for i in range(4)]
    return dtw_distances

# Функция рассчитывает DTW для каждого канала. DTW измеряет расстояние между временными рядами, учитывая возможность
# временной деформации (растягивание и сжатие по времени)
# Для расчета используется функция fastdtw, которая использует евклидово расстояние для измерения различий между
# значениями сигналов


# Расчет корреляций и DTW между всеми комбинациями сигналов
results = {}
for name1, data1 in data.items():
    for name2, data2 in data.items():
        if name1 != name2:
            pearson_corr, spearman_corr, kendall_corr = calculate_correlations(data1, data2)
            dtw_distances = calculate_dtw(data1, data2)
            results[f'{name1} vs {name2}'] = {
                'pearson': pearson_corr,
                'spearman': spearman_corr,
                'kendall': kendall_corr,
                'dtw': dtw_distances
            }

# Вывод результатов
for combo, result in results.items():
    print(f"Результаты для {combo}:")
    print(f"  Пирсон: {result['pearson']}")
    print(f"  Спирмен: {result['spearman']}")
    print(f"  Кендалл: {result['kendall']}")
    print(f"  DTW: {result['dtw']}")
    print()



import numpy as np
import matplotlib.pyplot as plt

# Функция для расчета спектральной плотности через FFT
def calculate_spectral_density(data):
    spectral_densities = []
    for i in range(4):  # Для каждого канала
        signal = data.iloc[:, i].to_numpy()  # Преобразуем данные в массив
        fft_values = np.fft.fft(signal)  # Применяем FFT
        fft_frequencies = np.fft.fftfreq(len(signal))  # Получаем частоты
        spectral_density = np.abs(fft_values) ** 2  # Вычисляем спектральную плотность
        spectral_densities.append((fft_frequencies, spectral_density))  # Сохраняем частоты и плотности
    return spectral_densities

# Здесь используется быстрое преобразование Фурье (FFT) для разложения сигналов по частотам. Результат отображается в
# виде спектральной плотности – это квадрат амплитуды спектра.

for name, data1 in data.items():
    spectral_density = calculate_spectral_density(data1)
    for i in range(4):
        freqs, density = spectral_density[i]
        plt.figure()
        plt.plot(freqs, density)
        plt.title(f"Спектральная плотность для {name}, Канал {i+1}")
        plt.xlabel('Частота')
        plt.ylabel('Спектральная плотность')
        plt.show()

# Спектральная плотность для сигнала из файла b30hz40, канал 1:
# На графике могут быть такие частоты:
# Пик на 10 Гц: Это может указывать на вибрации, вызванные регулярной работой подшипника. Это частота вращения.
# Пик на 30 Гц: Возможно, это вторичная гармоника, вызванная сложными механическими процессами или дефектами.
# Распределение мощности по высокочастотной области: Может указывать на присутствие шума в сигнале или на вибрации с очень высокой частотой.

# Спектральная плотность для сигнала из файла b30hz10, канал 2:
# Если ты видишь другой спектр (например, пик на другой частоте), это может указывать на изменение в динамике системы:
# Смещение пиков может свидетельствовать о том, что вибрации в системе изменились (например, из-за износа подшипников или других механических факторов).
# Отсутствие выраженных пиков: Если спектр "расплывчатый", это может указывать на нестабильность в системе или на то, что сигнал слишком зашумлен.



from statsmodels.tsa.seasonal import STL


# Функция для STL-разложения (сезонно-трендовое разложение)
def decompose_series(data):
    decompositions = []
    for i in range(4):  # Для каждого канала
        signal = data.iloc[:, i]  # Извлекаем данные одного канала
        stl = STL(signal, period=12)  # Выбираем период (можно настроить)
        result = stl.fit()
        decompositions.append(result)
    return decompositions

# STL-разложение делит временные ряды на три компоненты:
# Тренд (Trend): долгосрочная тенденция изменения сигнала
# Сезонность (Seasonal): периодические колебания или повторяющиеся циклы в сигнале
# Остаток (Residual): случайные колебания или шум, которые остаются после удаления тренда и сезонности


# для одного файла
for name, data1 in data.items():
    decompositions = decompose_series(data1)
    for i in range(4):
        result = decompositions[i]

        # Визуализация компонентов
        result.plot()
        plt.title(f"STL-разложение для {name}, Канал {i + 1}")
        plt.show()

# Тренд может показывать постепенное увеличение амплитуды вибраций, что указывает на ухудшение работы подшипника.
# Сезонность выявляет регулярные колебания (например, вибрации, которые появляются с определенной периодичностью в ходе работы системы).
# Остатки могут показывать резкие всплески вибраций, которые выходят за пределы сезонности и тренда, указывая на аномалии.
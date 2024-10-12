from scripts import data_preprocessing as dp
from scripts import visualization as vz
from scripts import feature_selection as fs
import pandas as pd

# Пусть к вашему CSV файлу
data_file = './data/Popular_Spotify_Songs.csv'

# 1. Предварительная обработка данных
df = dp.preprocess_data(data_file)

# Проверка данных
print("Форма датафрейма:", df.shape)
print("Первые 5 строк датафрейма:")
print(df.head())

# 2. Визуализация данных
# Тепловая карта корреляции
vz.plot_correlation_matrix(df)

# Гистограмма распределения
vz.plot_histogram(df, 'streams')  # Убедитесь, что 'streams' существует

# Scatter Plot: связи между 'streams' и 'bpm'
vz.plot_scatter(df, 'streams', 'bpm')  # Убедитесь, что 'bpm' существует

# 3. Отбор признаков
# Замените 'target_column' на фактическое имя вашего целевого столбца
X = df.drop(columns=['target_column'])  # Замените на нужный столбец
y = df['target_column']  # Замените на нужный целевой столбец

# Применим RFE для отбора признаков
selected_features = fs.recursive_feature_elimination(X, y, n_features=5)
print("Выбранные признаки:", X.columns[selected_features])

# Применение PCA
principal_components, variance_ratios = fs.apply_pca(X)
print("Principal components:", principal_components)
print("Explained variance ratio:", variance_ratios)

import pandas as pd
from sklearn.preprocessing import StandardScaler
import chardet

def detect_encoding(file_path):
    """Определяет кодировку файла."""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def load_data(file_path):
    """Загружает и возвращает DataFrame из CSV файла с определенной кодировкой."""
    encoding = detect_encoding(file_path)
    return pd.read_csv(file_path, encoding=encoding)

def handle_missing_values(df):
    """Обрабатывает пропущенные значения в наборе данных."""
    df = df.copy()  # Создаем копию
    df['key'] = df['key'].fillna('Unknown')
    df['in_shazam_charts'] = df['in_shazam_charts'].fillna(0)
    return df
# 
def normalize_data(df, numerical_columns):
    """Нормализует числовые признаки с использованием StandardScaler."""
    # Преобразуем все значения в числовой формат
    df[numerical_columns] = df[numerical_columns].apply(pd.to_numeric, errors='coerce')
    
    # Проверяем наличие NaN после преобразования
    print("Количество NaN после преобразования:", df[numerical_columns].isnull().sum())
    
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

def preprocess_data(file_path):
    """Выполняет полную предварительную обработку данных."""
    df = load_data(file_path)
    df = handle_missing_values(df)
    numerical_cols = ['streams', 'bpm', 'danceability_%', 'energy_%', 
                      'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']
    df = normalize_data(df, numerical_cols)
    return df

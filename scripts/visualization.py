import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(df):
    """Строит тепловую карту корреляционной матрицы."""
    # Выбираем только числовые столбцы
    numeric_df = df.select_dtypes(include='number')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def plot_histogram(df, column_name):
    """Строит гистограмму для заданного числового признака."""
    df[column_name].hist(bins=30)
    plt.title(f'Distribution of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.show()

def plot_scatter(df, x_col, y_col):
    """Строит scatter plot для двух признаков."""
    plt.scatter(df[x_col], df[y_col])
    plt.title(f'Scatter Plot: {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()

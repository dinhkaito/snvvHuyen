import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Импорт алгоритмов снижения размерности
from sklearn.manifold import TSNE
try:
    import umap
except ImportError:
    print("UMAP не установлен. Установите: pip install umap-learn")
try:
    import trimap
except ImportError:
    print("TriMAP не установлен. Установите: pip install trimap")
try:
    import pacmap
except ImportError:
    print("PaCMAP не установлен. Установите: pip install pacmap")

# Загрузка данных
def load_credit_approval_data():
    """Загрузка и предобработка данных Credit Approval"""
    # URL для скачивания данных
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
    
    # Названия колонок (анонимизированные)
    column_names = [f'A{i}' for i in range(1, 16)] + ['class']
    
    try:
        # Загрузка данных
        data = pd.read_csv(url, names=column_names, na_values='?')
        print(f"Данные успешно загружены. Размер: {data.shape}")
        return data
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        # Создание демо-данных для примера
        print("Создание демонстрационных данных...")
        np.random.seed(42)
        n_samples = 690
        data = pd.DataFrame(np.random.randn(n_samples, 15), 
                           columns=[f'A{i}' for i in range(1, 16)])
        data['class'] = np.random.choice(['+', '-'], n_samples)
        return data

# Предобработка данных
def preprocess_data(data):
    """Предобработка данных для визуализации"""
    # Создаем копию данных
    df = data.copy()
    
    # Кодируем целевую переменную
    le = LabelEncoder()
    df['class_encoded'] = le.fit_transform(df['class'])
    
    # Разделяем на признаки и целевую переменную
    X = df.drop(['class', 'class_encoded'], axis=1)
    y = df['class_encoded']
    
    # Обработка категориальных признаков
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    
    # Кодируем категориальные признаки
    for col in categorical_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    # Заполняем пропущенные значения
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Масштабируем данные
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    print(f"Данные предобработаны. Форма: {X_scaled.shape}")
    return X_scaled, y, le.classes_

# Функции для снижения размерности
def apply_tsne(X, perplexity=30):
    """Применение t-SNE"""
    print("Применение t-SNE...")
    tsne = TSNE(n_components=2, 
                perplexity=perplexity, 
                random_state=42,
                n_iter=1000)
    return tsne.fit_transform(X)

def apply_umap(X, n_neighbors=15):
    """Применение UMAP"""
    print("Применение UMAP...")
    reducer = umap.UMAP(n_components=2, 
                       n_neighbors=n_neighbors, 
                       random_state=42,
                       min_dist=0.1)
    return reducer.fit_transform(X)

def apply_trimap(X):
    """Применение TriMAP"""
    print("Применение TriMAP...")
    reducer = trimap.TRIMAP(n_dims=2, verbose=False)
    return reducer.fit_transform(X)

def apply_pacmap(X):
    """Применение PaCMAP"""
    print("Применение PaCMAP...")
    reducer = pacmap.PaCMAP(n_components=2, random_state=42)
    return reducer.fit_transform(X)

# Визуализация результатов
def plot_results(embeddings, y, class_names, titles, figsize=(20, 5)):
    """Визуализация результатов всех методов"""
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    for i, (embedding, title) in enumerate(zip(embeddings, titles)):
        scatter = axes[i].scatter(embedding[:, 0], embedding[:, 1], 
                                c=y, cmap='viridis', alpha=0.7, s=20)
        axes[i].set_title(title, fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Component 1')
        axes[i].set_ylabel('Component 2')
        axes[i].grid(alpha=0.3)
        
        # Добавляем цветовую легенду
        if i == 0:
            legend = axes[i].legend(*scatter.legend_elements(),
                                  title="Classes", loc='upper right')
    
    plt.tight_layout()
    plt.show()

# Сравнительный анализ
def compare_methods(embeddings, titles, y):
    """Сравнительный анализ методов"""
    from sklearn.metrics import silhouette_score
    
    print("\n" + "="*50)
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ МЕТОДОВ")
    print("="*50)
    
    results = []
    for i, (embedding, title) in enumerate(zip(embeddings, titles)):
        # Вычисляем силуэтный коэффициент
        sil_score = silhouette_score(embedding, y)
        
        # Вычисляем дисперсию
        variance_x = np.var(embedding[:, 0])
        variance_y = np.var(embedding[:, 1])
        total_variance = variance_x + variance_y
        
        results.append({
            'Method': title,
            'Silhouette Score': sil_score,
            'Total Variance': total_variance,
            'Variance X': variance_x,
            'Variance Y': variance_y
        })
        
        print(f"\n{title}:")
        print(f"  Silhouette Score: {sil_score:.4f}")
        print(f"  Total Variance: {total_variance:.4f}")
        print(f"  Variance X: {variance_x:.4f}")
        print(f"  Variance Y: {variance_y:.4f}")
    
    return pd.DataFrame(results)

# Основная функция
def main():
    """Основная функция выполнения визуализации"""
    print("ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ")
    print("="*50)
    
    # Загрузка данных
    data = load_credit_approval_data()
    
    # Предобработка данных
    X_scaled, y, class_names = preprocess_data(data)
    
    print(f"\nКлассы: {class_names}")
    print(f"Распределение классов: {pd.Series(y).value_counts().to_dict()}")
    
    # Применение алгоритмов снижения размерности
    print("\nПРИМЕНЕНИЕ АЛГОРИТМОВ СНИЖЕНИЯ РАЗМЕРНОСТИ")
    print("="*50)
    
    embeddings = []
    titles = []
    
    # t-SNE
    try:
        X_tsne = apply_tsne(X_scaled)
        embeddings.append(X_tsne)
        titles.append('t-SNE')
    except Exception as e:
        print(f"Ошибка в t-SNE: {e}")
    
    # UMAP
    try:
        X_umap = apply_umap(X_scaled)
        embeddings.append(X_umap)
        titles.append('UMAP')
    except Exception as e:
        print(f"Ошибка в UMAP: {e}")
    
    # TriMAP
    try:
        X_trimap = apply_trimap(X_scaled)
        embeddings.append(X_trimap)
        titles.append('TriMAP')
    except Exception as e:
        print(f"Ошибка в TriMAP: {e}")
    
    # PaCMAP
    try:
        X_pacmap = apply_pacmap(X_scaled)
        embeddings.append(X_pacmap)
        titles.append('PaCMAP')
    except Exception as e:
        print(f"Ошибка в PaCMAP: {e}")
    
    # Визуализация результатов
    print("\nВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("="*50)
    
    plot_results(embeddings, y, class_names, titles)
    
    # Сравнительный анализ
    results_df = compare_methods(embeddings, titles, y)
    
    # Дополнительная визуализация - все методы вместе
    print("\nСОВМЕСТНАЯ ВИЗУАЛИЗАЦИЯ")
    print("="*50)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, (embedding, title) in enumerate(zip(embeddings, titles)):
        scatter = axes[i].scatter(embedding[:, 0], embedding[:, 1], 
                                c=y, cmap='Set2', alpha=0.7, s=30)
        axes[i].set_title(f'{title}\n(Silhouette: {results_df.iloc[i]["Silhouette Score"]:.3f})', 
                         fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Component 1')
        axes[i].set_ylabel('Component 2')
        axes[i].grid(alpha=0.3)
    
    # Убираем лишние subplots
    for i in range(len(embeddings), 4):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()
    
    return results_df, embeddings, titles

# Запуск анализа
if __name__ == "__main__":
    results_df, embeddings, titles = main()
    
    # Вывод итоговой таблицы
    print("\n" + "="*50)
    print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*50)
    print(results_df.round(4))
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import pickle
import warnings
import os
from ast import literal_eval

warnings.filterwarnings('ignore')


class MovieClustering:
    """
    Класс для кластеризации фильмов по различным показателям.
    Методы: Агломеративная кластеризация (Ward), k-means, DBSCAN
    """

    def __init__(self, movies_path, credits_path=None):
        """Инициализация системы кластеризации"""
        print(f"\n📂 Загрузка данных для кластеризации...")
        self.movies_df = pd.read_csv(movies_path)
        self.credits_df = pd.read_csv(credits_path) if credits_path else None

        self.data_prepared = None
        self.data_standardized = None
        self.feature_columns = None
        self.scaler = None
        self.movies_labels = None

        # Модели кластеризации
        self.hierarchical_model = None
        self.kmeans_models = {}  # {n_clusters: model}
        self.dbscan_model = None

        # Результаты кластеризации
        self.hierarchical_labels = None
        self.kmeans_labels = {}
        self.dbscan_labels = None

        # Метрики
        self.hierarchical_metrics = None
        self.kmeans_metrics = {}
        self.dbscan_metrics = None

        print(f"✅ Загружено фильмов: {len(self.movies_df)}")
        if self.credits_df is not None:
            print(f"✅ Загружена информация об актерах: {len(self.credits_df)} записей")

    def _extract_cast_count(self, movie_id):
        """Получить количество актеров из credits"""
        if self.credits_df is None:
            return 0

        try:
            row = self.credits_df[self.credits_df['id'] == movie_id]
            if len(row) > 0:
                cast_str = row.iloc[0]['cast']
                if isinstance(cast_str, str) and cast_str.strip():
                    cast = literal_eval(cast_str)
                    if isinstance(cast, list):
                        return len(cast)
        except Exception:
            pass
        return 0

    def _extract_main_genre(self, genres_str):
        """Извлечение основного жанра"""
        try:
            if isinstance(genres_str, str) and genres_str != '[]':
                genres_list = literal_eval(genres_str)
                if isinstance(genres_list, list) and len(genres_list) > 0:
                    return genres_list[0].get('name', 'Unknown')
        except Exception:
            pass
        return 'Unknown'

    def prepare_data(self):
        """Подготовка данных для кластеризации"""
        print("\n" + "=" * 70)
        print("📊 ПОДГОТОВКА ДАННЫХ ДЛЯ КЛАСТЕРИЗАЦИИ")
        print("=" * 70)

        df = self.movies_df.copy()

        # Конвертируем в числовые типы
        df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
        df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')
        df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce')
        df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
        df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')

        # Фильтруем: оставляем только фильмы с достаточной информацией
        df = df[
            (df['vote_count'] > 0) &
            (df['vote_average'] > 0) &
            (df['popularity'] > 0) &
            (df['runtime'] > 0)
            ].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError("❌ Недостаточно данных для кластеризации")

        print(f"\n✅ Фильмов для кластеризации: {len(df)}")

        # Извлекаем признаки
        print("🔧 Извлечение признаков...")

        # Основные показатели
        features = pd.DataFrame()
        features['vote_average'] = df['vote_average'].fillna(0)
        features['vote_count'] = df['vote_count'].fillna(0)
        features['runtime'] = df['runtime'].fillna(0)
        features['popularity'] = df['popularity'].fillna(0)
        features['budget'] = df['budget'].fillna(0)
        features['revenue'] = df['revenue'].fillna(0)

        # Логарифмические признаки
        features['log_vote_count'] = np.log1p(features['vote_count'])
        features['log_popularity'] = np.log1p(features['popularity'])
        features['log_budget'] = np.log1p(features['budget'])
        features['log_revenue'] = np.log1p(features['revenue'])

        # Извлекаем количество актеров
        print("🎭 Извлечение количества актеров...")
        features['cast_count'] = df['id'].apply(self._extract_cast_count)
        features['log_cast_count'] = np.log1p(features['cast_count'])

        # Отношения признаков (инженерия признаков)
        features['vote_per_cast'] = np.divide(
            features['vote_count'],
            features['cast_count'],
            where=features['cast_count'] != 0,
            out=np.zeros_like(features['vote_count'])
        )

        features['revenue_per_budget'] = np.divide(
            features['revenue'],
            features['budget'],
            where=features['budget'] != 0,
            out=np.zeros_like(features['revenue'])
        )

        features['budget_per_runtime'] = np.divide(
            features['budget'],
            features['runtime'],
            where=features['runtime'] != 0,
            out=np.zeros_like(features['budget'])
        )

        features['profit'] = features['revenue'] - features['budget']
        features['roi'] = np.divide(
            features['profit'],
            features['budget'],
            where=features['budget'] != 0,
            out=np.zeros_like(features['profit'])
        )

        # Сохраняем подготовленные данные
        self.data_prepared = features
        self.movies_labels = df['title'].tolist()
        self.feature_columns = features.columns.tolist()

        print(f"✅ Используется {len(self.feature_columns)} признаков")
        print(f"   Признаки: {self.feature_columns}")
        print(f"\n✅ Статистика признаков:")
        print(features.describe().round(2))

        return features

    def standardize_data(self):
        """Стандартизация данных"""
        if self.data_prepared is None:
            raise ValueError("Сначала подготовьте данные (prepare_data)")

        print("\n" + "-" * 70)
        print("🔧 Стандартизация данных...")

        self.scaler = StandardScaler()
        self.data_standardized = self.scaler.fit_transform(self.data_prepared)

        print("✅ Данные стандартизированы")

        return self.data_standardized

    def hierarchical_clustering(self, n_clusters=5):
        """
        Агломеративная кластеризация (Ward's method)

        Parameters:
        -----------
        n_clusters : int
            Количество кластеров
        """
        print("\n" + "=" * 70)
        print("🏗️ ИЕРАРХИЧЕСКАЯ КЛАСТЕРИЗАЦИЯ (WARD'S METHOD)")
        print("=" * 70)

        if self.data_standardized is None:
            self.prepare_data()
            self.standardize_data()

        print(f"\n📊 Кластеризация на {n_clusters} классов...")

        # Агломеративная кластеризация
        self.hierarchical_model = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity='euclidean',
            linkage='ward'
        )

        self.hierarchical_labels = self.hierarchical_model.fit_predict(self.data_standardized)

        # Расчет метрик
        silhouette = silhouette_score(self.data_standardized, self.hierarchical_labels)

        print(f"\n✅ Кластеризация завершена")
        print(f"   Silhouette Score: {silhouette:.3f}")

        # Распределение по кластерам
        unique, counts = np.unique(self.hierarchical_labels, return_counts=True)
        print(f"\n📊 Распределение по кластерам:")
        for cluster_id, count in zip(unique, counts):
            percentage = (count / len(self.hierarchical_labels)) * 100
            print(f"   Кластер {cluster_id}: {count} фильмов ({percentage:.1f}%)")

        self.hierarchical_metrics = {
            'n_clusters': n_clusters,
            'silhouette_score': float(silhouette),
            'cluster_distribution': dict(zip(unique.tolist(), counts.tolist()))
        }

        return self.hierarchical_labels

    def kmeans_clustering(self, n_clusters=5, n_init=10, max_iter=300):
        """
        Кластеризация методом k-means

        Parameters:
        -----------
        n_clusters : int
            Количество кластеров
        n_init : int
            Количество инициализаций
        max_iter : int
            Максимальное количество итераций
        """
        print("\n" + "=" * 70)
        print("⭐ КЛАСТЕРИЗАЦИЯ k-MEANS")
        print("=" * 70)

        if self.data_standardized is None:
            self.prepare_data()
            self.standardize_data()

        print(f"\n📊 Кластеризация на {n_clusters} классов...")
        print(f"   Параметры: n_init={n_init}, max_iter={max_iter}")

        # k-means
        kmeans = KMeans(
            n_clusters=n_clusters,
            init='random',
            n_init=n_init,
            max_iter=max_iter,
            tol=1e-04,
            random_state=0
        )

        labels = kmeans.fit_predict(self.data_standardized)

        # Расчет метрик
        silhouette = silhouette_score(self.data_standardized, labels)
        inertia = kmeans.inertia_

        print(f"\n✅ Кластеризация завершена")
        print(f"   Silhouette Score: {silhouette:.3f}")
        print(f"   Inertia (сумма квадратичных расстояний): {inertia:.2f}")

        # Распределение по кластерам
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n📊 Распределение по кластерам:")
        for cluster_id, count in zip(unique, counts):
            percentage = (count / len(labels)) * 100
            print(f"   Кластер {cluster_id}: {count} фильмов ({percentage:.1f}%)")

        # Сохраняем результаты
        self.kmeans_models[n_clusters] = kmeans
        self.kmeans_labels[n_clusters] = labels
        self.kmeans_metrics[n_clusters] = {
            'n_clusters': n_clusters,
            'silhouette_score': float(silhouette),
            'inertia': float(inertia),
            'cluster_distribution': dict(zip(unique.tolist(), counts.tolist()))
        }

        return labels

    def dbscan_clustering(self, eps=0.5, min_samples=5):
        """
        Кластеризация DBSCAN

        Parameters:
        -----------
        eps : float
            Радиус окрестности
        min_samples : int
            Минимальное количество образцов в окрестности
        """
        print("\n" + "=" * 70)
        print("🔍 КЛАСТЕРИЗАЦИЯ DBSCAN")
        print("=" * 70)

        if self.data_standardized is None:
            self.prepare_data()
            self.standardize_data()

        print(f"\n📊 Кластеризация DBSCAN...")
        print(f"   Параметры: eps={eps}, min_samples={min_samples}")

        # DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        self.dbscan_labels = dbscan.fit_predict(self.data_standardized)

        # Расчет метрик
        n_clusters = len(set(self.dbscan_labels)) - (1 if -1 in self.dbscan_labels else 0)
        n_noise = list(self.dbscan_labels).count(-1)

        # Silhouette score (только если есть хотя бы 2 кластера и нет шума)
        if n_clusters > 1 and n_noise == 0:
            silhouette = silhouette_score(self.data_standardized, self.dbscan_labels)
        else:
            silhouette = None

        print(f"\n✅ Кластеризация завершена")
        print(f"   Количество кластеров: {n_clusters}")
        print(f"   Количество точек шума: {n_noise}")
        if silhouette is not None:
            print(f"   Silhouette Score: {silhouette:.3f}")

        # Распределение
        unique, counts = np.unique(self.dbscan_labels, return_counts=True)
        print(f"\n📊 Распределение:")
        for cluster_id, count in zip(unique, counts):
            if cluster_id == -1:
                print(f"   Шум: {count} фильмов ({count / len(self.dbscan_labels) * 100:.1f}%)")
            else:
                print(f"   Кластер {cluster_id}: {count} фильмов ({count / len(self.dbscan_labels) * 100:.1f}%)")

        self.dbscan_metrics = {
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise': int(n_noise),
            'silhouette_score': float(silhouette) if silhouette is not None else None,
            'cluster_distribution': dict(zip([int(x) for x in unique], counts.tolist()))
        }

        return self.dbscan_labels

    def get_cluster_profiles(self, labels, n_clusters=None):
        """
        Получить профили кластеров (средние значения признаков)

        Parameters:
        -----------
        labels : array
            Метки кластеров
        n_clusters : int
            Количество кластеров (автоматически определяется, если None)
        """
        if n_clusters is None:
            n_clusters = len(set(labels))

        profiles = []

        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_data = self.data_prepared[cluster_mask]

            profile = {
                'cluster_id': cluster_id,
                'size': cluster_mask.sum(),
                'means': cluster_data.mean().to_dict(),
                'medians': cluster_data.median().to_dict(),
                'stds': cluster_data.std().to_dict()
            }

            profiles.append(profile)

        return profiles

    def get_cluster_movies(self, labels, cluster_id):
        """
        Получить фильмы из определенного кластера

        Parameters:
        -----------
        labels : array
            Метки кластеров
        cluster_id : int
            ID кластера

        Returns:
        --------
        list
            Список названий фильмов в кластере
        """
        cluster_mask = labels == cluster_id
        movie_indices = np.where(cluster_mask)[0]

        movies = [self.movies_labels[idx] for idx in movie_indices]

        return movies

    def save_clustering(self, filepath='movie_clustering_model.pkl'):
        """Сохранение моделей кластеризации"""
        data = {
            'data_prepared': self.data_prepared,
            'data_standardized': self.data_standardized,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'movies_labels': self.movies_labels,
            'hierarchical_model': self.hierarchical_model,
            'hierarchical_labels': self.hierarchical_labels,
            'hierarchical_metrics': self.hierarchical_metrics,
            'kmeans_models': self.kmeans_models,
            'kmeans_labels': self.kmeans_labels,
            'kmeans_metrics': self.kmeans_metrics,
            'dbscan_model': self.dbscan_model,
            'dbscan_labels': self.dbscan_labels,
            'dbscan_metrics': self.dbscan_metrics
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"\n💾 Модель кластеризации сохранена: {filepath}")

    def load_clustering(self, filepath='movie_clustering_model.pkl'):
        """Загрузка моделей кластеризации"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.data_prepared = data['data_prepared']
        self.data_standardized = data['data_standardized']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.movies_labels = data['movies_labels']
        self.hierarchical_model = data['hierarchical_model']
        self.hierarchical_labels = data['hierarchical_labels']
        self.hierarchical_metrics = data['hierarchical_metrics']
        self.kmeans_models = data['kmeans_models']
        self.kmeans_labels = data['kmeans_labels']
        self.kmeans_metrics = data['kmeans_metrics']
        self.dbscan_model = data['dbscan_model']
        self.dbscan_labels = data['dbscan_labels']
        self.dbscan_metrics = data['dbscan_metrics']

        print(f"\n✅ Модель кластеризации загружена: {filepath}")


# ============ ДЕМОНСТРАЦИЯ И ТЕСТИРОВАНИЕ ============

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🎬 СИСТЕМА КЛАСТЕРИЗАЦИИ ФИЛЬМОВ")
    print("=" * 70)

    # Пути к файлам - ИЗМЕНИТЕ НА СВОИ ПУТИ!
    movies_path = r"C:\Users\Даниил\Downloads\archive\movies_metadata.csv"
    credits_path = r"C:\Users\Даниил\Downloads\archive\credits.csv"

    # Проверяем существование файлов
    print("\n🔍 Проверка файлов данных:")
    if os.path.exists(movies_path):
        print(f"  ✅ movies_metadata.csv найден")
    else:
        print(f"  ❌ movies_metadata.csv НЕ НАЙДЕН")
        exit(1)

    if os.path.exists(credits_path):
        print(f"  ✅ credits.csv найден")
    else:
        print(f"  ❌ credits.csv НЕ НАЙДЕН")

    # Инициализация системы кластеризации
    clustering = MovieClustering(movies_path, credits_path)

    # Подготовка данных
    clustering.prepare_data()
    clustering.standardize_data()

    # ========== ИЕРАРХИЧЕСКАЯ КЛАСТЕРИЗАЦИЯ ==========
    print("\n" + "#" * 70)
    hierarchical_labels = clustering.hierarchical_clustering(n_clusters=5)

    # Получить профили кластеров
    hierarchical_profiles = clustering.get_cluster_profiles(hierarchical_labels, n_clusters=5)
    print("\n📈 Средние значения признаков для иерархической кластеризации:")
    for profile in hierarchical_profiles:
        print(f"\n🎬 Кластер {profile['cluster_id']} ({profile['size']} фильмов):")
        for feature, value in list(profile['means'].items())[:5]:  # Показываем первые 5 признаков
            print(f"   {feature}: {value:.2f}")

    # Примеры фильмов из кластеров
    print("\n🎞️ Примеры фильмов из иерархических кластеров:")
    for cluster_id in range(5):
        movies = clustering.get_cluster_movies(hierarchical_labels, cluster_id)
        print(f"\n   Кластер {cluster_id}: {movies[:3]}...")  # Показываем первые 3 фильма

    # ========== k-MEANS КЛАСТЕРИЗАЦИЯ ==========
    print("\n" + "#" * 70)
    kmeans_labels = clustering.kmeans_clustering(n_clusters=5)

    # Получить профили кластеров
    kmeans_profiles = clustering.get_cluster_profiles(kmeans_labels, n_clusters=5)
    print("\n📈 Средние значения признаков для k-means:")
    for profile in kmeans_profiles:
        print(f"\n🎬 Кластер {profile['cluster_id']} ({profile['size']} фильмов):")
        for feature, value in list(profile['means'].items())[:5]:
            print(f"   {feature}: {value:.2f}")

    # ========== DBSCAN КЛАСТЕРИЗАЦИЯ ==========
    print("\n" + "#" * 70)
    dbscan_labels = clustering.dbscan_clustering(eps=1.5, min_samples=5)

    # Сохранение модели
    clustering.save_clustering('movie_clustering_model.pkl')

    print("\n" + "=" * 70)
    print("✅ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 70)

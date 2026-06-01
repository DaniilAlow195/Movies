import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from ast import literal_eval
import warnings
import pickle
import os

warnings.filterwarnings('ignore')


class RevenuePredictor:
    """Предсказатель доходов фильмов с использованием Gradient Boosting"""
    
    def __init__(self, movies_path, credits_path=None):
        """Инициализация предсказателя доходов"""
        self.movies_df = pd.read_csv(movies_path)
        self.credits_df = pd.read_csv(credits_path) if credits_path else None

        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.label_encoders = {}
        self.prepared_data = None
        self.metrics = {}

        # Диагностика
        print(f"\n📊 Диагностика данных для предсказателя доходов:")
        print(f"   Фильмов в movies_metadata.csv: {len(self.movies_df)}")
        print(f"   Колонки: {list(self.movies_df.columns[:10])}")
        if self.credits_df is not None:
            print(f"   Строк в credits.csv: {len(self.credits_df)}")
            print(f"   Колонки: {list(self.credits_df.columns)}")

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

    def _extract_cast_count(self, movie_id):
        """Получить количество актеров из credits"""
        if self.credits_df is None:
            return 0

        try:
            # Ищем в credits_df по ID
            row = self.credits_df[self.credits_df['id'] == movie_id]
            if len(row) > 0:
                cast_str = row.iloc[0]['cast']
                if isinstance(cast_str, str) and cast_str.strip():
                    cast = literal_eval(cast_str)
                    if isinstance(cast, list):
                        return len(cast)
        except Exception as e:
            pass
        return 0

    def _prepare_data(self):
        """Подготовка данных для обучения"""
        df = self.movies_df.copy()

        # Конвертируем в числовые типы
        df['budget']    = pd.to_numeric(df['budget'], errors='coerce')
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
        df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')
        df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce')
        df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
        df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')

        # Фильтруем: оставляем только фильмы с доходом и бюджетом > 0
        df = df[(df['revenue'] > 0) & (df['budget'] > 0)].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError("Недостаточно данных с ненулевыми доходом и бюджетом")

        print(f"\n📊 Фильмов для обучения (с бюджетом и доходом): {len(df)}")

        # Извлекаем признаки
        df['main_genre'] = df['genres'].apply(self._extract_main_genre)
        df['original_language'] = df['original_language'].fillna('en')

        # Извлекаем количество актеров
        print("🎭 Извлечение количества актеров из credits...")
        df['cast_count'] = df['id'].apply(self._extract_cast_count)

        # Проверяем результат
        cast_not_zero = (df['cast_count'] > 0).sum()
        print(f"   ✅ Фильмов с информацией об актерах: {cast_not_zero}/{len(df)}")

        if cast_not_zero == 0:
            print("   ⚠️  ВНИМАНИЕ! Ни один фильм не имеет информации об актерах!")
            print("   Возможно, ID в movies_metadata.csv и credits.csv не совпадают.")

        # Создаем DataFrame признаков
        features = pd.DataFrame()
        features['budget'] = df['budget']
        features['vote_average'] = df['vote_average'].fillna(0)
        features['vote_count'] = df['vote_count'].fillna(0)
        features['runtime'] = df['runtime'].fillna(0)
        features['popularity'] = df['popularity'].fillna(0)
        features['cast_count'] = df['cast_count'].fillna(0)

        # Логарифмические признаки
        features['log_budget'] = np.log1p(features['budget'])
        features['log_vote_count'] = np.log1p(features['vote_count'])
        features['budget_per_runtime'] = np.divide(
            features['budget'],
            features['runtime'],
            where=features['runtime'] != 0,
            out=np.zeros_like(features['budget'])
        )

        # Кодируем категориальные переменные
        for col in ['main_genre', 'original_language']:
            le = LabelEncoder()
            features[col + '_encoded'] = le.fit_transform(df[col].fillna('unknown'))
            self.label_encoders[col] = le

        target = df['revenue']

        self.prepared_data = {
            'features': features,
            'target': target,
            'df': df
        }

        return features, target

    def train_model(self, test_size=0.2, random_state=42):
        """Обучение модели"""
        print("\n📊 Подготовка данных...")
        features, target = self._prepare_data()

        self.feature_columns = features.columns.tolist()

        print(f"✅ Используется {len(self.feature_columns)} признаков")
        print(f"   Признаки: {self.feature_columns}")

        # Разделяем данные
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state
        )

        # Масштабируем
        self.scaler = StandardScaler()

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Создаем модель
        print(f"\n🚀 Обучение Gradient Boosting модели...")

        self.model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            verbose=0,
            subsample=0.8
        )

        # Обучение
        self.model.fit(X_train_scaled, y_train)

        # Предсказания
        y_pred = self.model.predict(X_test_scaled)

        # Метрики
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"\n📈 Результаты модели:")
        print(f"   R² Score: {r2:.4f}")
        print(f"   MAE: ${mae:,.0f}")
        print(f"   MSE: ${mse:,.0f}")
        print(f"   RMSE: ${rmse:,.0f}")

        # Важность признаков
        importances = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n⭐ Топ-5 важных признаков:")

        for idx, row in importances.head(5).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")

        # СОХРАНЯЕМ МЕТРИКИ
        self.metrics = {
            'r2': float(r2),
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'model_type': 'Gradient Boosting Regressor'
        }

        return self.metrics

    def find_movie_by_title(self, title):
        """Поиск фильма в датасете по названию"""
        title_lower = title.lower().strip()

        # Точный поиск
        exact_match = self.movies_df[self.movies_df['title'].str.lower() == title_lower]
        if len(exact_match) > 0:
            return exact_match.iloc[0]

        # Частичный поиск
        partial_match = self.movies_df[
            self.movies_df['title'].str.lower().str.contains(title_lower, na=False, regex=False)
        ]

        if len(partial_match) > 0:
            return partial_match.iloc[0]

        return None

    def predict_by_title(self, movie_title):
        """Предсказание дохода по названию фильма"""
        if self.model is None:
            raise ValueError("Модель не обучена")

        # Ищем фильм в датасете
        movie = self.find_movie_by_title(movie_title)

        if movie is None:
            return None

        # Извлекаем признаки
        budget = float(movie['budget']) if pd.notna(movie['budget']) else 0
        vote_average = float(movie['vote_average']) if pd.notna(movie['vote_average']) else 0
        vote_count = float(movie['vote_count']) if pd.notna(movie['vote_count']) else 0
        runtime = float(movie['runtime']) if pd.notna(movie['runtime']) else 0
        popularity = float(movie['popularity']) if pd.notna(movie['popularity']) else 0
        main_genre = self._extract_main_genre(movie['genres'])
        original_language = str(movie['original_language']) if pd.notna(movie['original_language']) else 'en'

        # Получаем количество актеров
        cast_count = self._extract_cast_count(movie['id'])

        # Подготавливаем признаки
        features = pd.DataFrame({
            'budget': [budget],
            'vote_average': [vote_average],
            'vote_count': [vote_count],
            'runtime': [runtime],
            'popularity': [popularity],
            'cast_count': [cast_count],
            'log_budget': [np.log1p(budget)],
            'log_vote_count': [np.log1p(vote_count)],
            'budget_per_runtime': [budget / runtime if runtime > 0 else 0]
        })

        # Кодируем категориальные переменные
        if main_genre in self.label_encoders['main_genre'].classes_:
            genre_code = self.label_encoders['main_genre'].transform([main_genre])[0]
        else:
            genre_code = self.label_encoders['main_genre'].transform(['Unknown'])[0]
        features['main_genre_encoded'] = genre_code

        if original_language in self.label_encoders['original_language'].classes_:
            lang_code = self.label_encoders['original_language'].transform([original_language])[0]
        else:
            lang_code = self.label_encoders['original_language'].transform(['en'])[0]
        features['original_language_encoded'] = lang_code

        # Упорядочиваем
        features = features[self.feature_columns]

        # Масштабируем и предсказываем
        features_scaled = self.scaler.transform(features)
        predicted_revenue = self.model.predict(features_scaled)[0]

        # Получаем актуальный доход из датасета
        actual_revenue = float(movie['revenue']) if pd.notna(movie['revenue']) else 0

        return {
            'title': movie['title'],
            'release_date': str(movie['release_date'])[:4] if pd.notna(movie['release_date']) else 'N/A',
            'budget': budget,
            'predicted_revenue': max(0, predicted_revenue),
            'actual_revenue': actual_revenue,
            'vote_average': vote_average,
            'popularity': popularity,
            'runtime': runtime,
            'cast_count': cast_count,
            'genres': self._extract_main_genre(movie['genres']),
            'vote_count': vote_count,
            'original_language': original_language
        }

    def save_model(self, filepath='revenue_model.pkl'):
        """Сохранение модели"""

        if self.model is None:
            raise ValueError("Нет обученной модели")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'label_encoders': self.label_encoders,
            'metrics': self.metrics
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\n💾 Модель сохранена: {filepath}")

    def load_model(self, filepath='revenue_model.pkl'):
        """Загрузка модели"""

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.label_encoders = model_data['label_encoders']

        # Загружаем метрики
        self.metrics = model_data.get('metrics', {
            'r2': 0,
            'mae': 0,
            'mse': 0,
            'rmse': 0,
            'model_type': 'Gradient Boosting Regressor'
        })

        print(f"✅ Модель загружена: {filepath}")

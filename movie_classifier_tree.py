import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
import warnings
import pickle
import os
from ast import literal_eval

warnings.filterwarnings('ignore')


class MovieClassifierTree:
    """
    Классификатор фильмов для определения успешности/популярности фильма
    используя ДЕРЕВО КЛАССИФИКАЦИИ (Decision Tree)

    Классификация: 0 - Низкая успешность, 1 - Высокая успешность
    На основе: голосов, рейтинга, популярности, бюджета и других факторов
    """

    def __init__(self, movies_path, credits_path=None):
        """Инициализация классификатора с деревом решений"""
        self.movies_df = pd.read_csv(movies_path)
        self.credits_df = pd.read_csv(credits_path) if credits_path else None

        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.label_encoders = {}
        self.prepared_data = None
        self.metrics = None

        print(f"\n📊 Диагностика данных для классификатора:")
        print(f"   Фильмов в movies_metadata.csv: {len(self.movies_df)}")
        print(f"   Колонки: {list(self.movies_df.columns[:10])}")
        if self.credits_df is not None:
            print(f"   Строк в credits.csv: {len(self.credits_df)}")

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

    def _prepare_data(self):
        """Подготовка данных для обучения"""
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
            (df['popularity'] > 0)
            ].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError("Недостаточно данных для обучения")

        print(f"\n📊 Фильмов для обучения: {len(df)}")

        # ЦЕЛЕВАЯ ПЕРЕМЕННАЯ: Классификация по успешности
        # Успешный фильм = vote_average >= 6.5 И vote_count >= медиана
        vote_count_median = df['vote_count'].median()
        df['is_successful'] = (
                (df['vote_average'] >= 6.5) &
                (df['vote_count'] >= vote_count_median)
        ).astype(int)

        successful_count = (df['is_successful'] == 1).sum()
        unsuccessful_count = (df['is_successful'] == 0).sum()

        print(f"   ✅ Успешные фильмы (vote_avg>=6.5 И vote_count>=медиана): {successful_count}")
        print(f"   ❌ Неуспешные фильмы: {unsuccessful_count}")
        print(f"   Распределение: {successful_count}/{len(df)} = {successful_count / len(df) * 100:.1f}%")

        # Извлекаем признаки
        df['main_genre'] = df['genres'].apply(self._extract_main_genre)
        df['original_language'] = df['original_language'].fillna('en')

        # Извлекаем количество актеров
        print("🎭 Извлечение количества актеров из credits...")
        df['cast_count'] = df['id'].apply(self._extract_cast_count)

        cast_not_zero = (df['cast_count'] > 0).sum()
        print(f"   ✅ Фильмов с информацией об актерах: {cast_not_zero}/{len(df)}")

        # Создаем DataFrame признаков
        features = pd.DataFrame()
        features['vote_average'] = df['vote_average'].fillna(0)
        features['vote_count'] = df['vote_count'].fillna(0)
        features['runtime'] = df['runtime'].fillna(0)
        features['popularity'] = df['popularity'].fillna(0)
        features['cast_count'] = df['cast_count'].fillna(0)
        features['budget'] = df['budget'].fillna(0)

        # Логарифмические признаки
        features['log_vote_count'] = np.log1p(features['vote_count'])
        features['log_popularity'] = np.log1p(features['popularity'])
        features['log_budget'] = np.log1p(features['budget'])

        # Отношения признаков
        features['vote_per_cast'] = np.divide(
            features['vote_count'],
            features['cast_count'],
            where=features['cast_count'] != 0,
            out=np.zeros_like(features['vote_count'])
        )

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

        target = df['is_successful']

        self.prepared_data = {
            'features': features,
            'target': target,
            'df': df
        }

        return features, target

    def train_model(self, test_size=0.2, random_state=42, max_depth=4,
                    min_samples_split=5, min_samples_leaf=2):
        """Обучение дерева классификации"""
        print("\n" + "=" * 60)
        print("🎬 ОБУЧЕНИЕ ДЕРЕВА КЛАССИФИКАЦИИ ФИЛЬМОВ")
        print("=" * 60)

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

        # Масштабируем (для консистентности, хотя Decision Tree его не требует)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # ====== ОБУЧЕНИЕ ДЕРЕВА КЛАССИФИКАЦИИ ======
        print("\n" + "-" * 60)
        print("🌳 ДЕРЕВО КЛАССИФИКАЦИИ (Decision Tree)")
        print("-" * 60)
        print(f"Параметры:")
        print(f"   max_depth: {max_depth}")
        print(f"   min_samples_split: {min_samples_split}")
        print(f"   min_samples_leaf: {min_samples_leaf}")

        # Создаем модель дерева
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )

        # Обучение модели на тренировочных данных
        self.model.fit(X_train_scaled, y_train)

        # Предсказания
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)

        # ====== МЕТРИКИ НА ТРЕНИРОВОЧНЫХ ДАННЫХ ======
        accuracy_train = accuracy_score(y_train, y_pred_train)
        precision_train = precision_score(y_train, y_pred_train)
        recall_train = recall_score(y_train, y_pred_train)
        f1_train = f1_score(y_train, y_pred_train)
        roc_auc_train = roc_auc_score(y_train, y_pred_train)

        confmat_train = confusion_matrix(y_train, y_pred_train)

        print(f"\n📈 Метрики на ТРЕНИРОВОЧНЫХ данных:")
        print(f"   Accuracy (правильность): {accuracy_train:.3f}")
        print(f"   Precision (точность): {precision_train:.3f}")
        print(f"   Recall (полнота): {recall_train:.3f}")
        print(f"   F1-score: {f1_train:.3f}")
        print(f"   ROC AUC: {roc_auc_train:.3f}")
        print(f"\n   Матрица ошибок:")
        print(f"   [[TN={confmat_train[0, 0]} FP={confmat_train[0, 1]}]")
        print(f"    [FN={confmat_train[1, 0]} TP={confmat_train[1, 1]}]]")

        # ====== МЕТРИКИ НА ТЕСТОВЫХ ДАННЫХ ======
        accuracy_test = accuracy_score(y_test, y_pred_test)
        precision_test = precision_score(y_test, y_pred_test)
        recall_test = recall_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test)
        roc_auc_test = roc_auc_score(y_test, y_pred_test)

        confmat_test = confusion_matrix(y_test, y_pred_test)

        print(f"\n📈 Метрики на ТЕСТОВЫХ данных:")
        print(f"   Accuracy (правильность): {accuracy_test:.3f}")
        print(f"   Precision (точность): {precision_test:.3f}")
        print(f"   Recall (полнота): {recall_test:.3f}")
        print(f"   F1-score: {f1_test:.3f}")
        print(f"   ROC AUC: {roc_auc_test:.3f}")
        print(f"\n   Матрица ошибок:")
        print(f"   [[TN={confmat_test[0, 0]} FP={confmat_test[0, 1]}]")
        print(f"    [FN={confmat_test[1, 0]} TP={confmat_test[1, 1]}]]")

        # Важность признаков
        importances = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n⭐ Топ-5 значимых признаков:")
        for idx, row in importances.head(5).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")

        # ✅ СОХРАНЯЕМ ТЕСТОВЫЕ МЕТРИКИ (не тренировочные!)
        self.metrics = {
            'accuracy': float(accuracy_test),
            'precision': float(precision_test),
            'recall': float(recall_test),
            'f1': float(f1_test),
            'roc_auc': float(roc_auc_test),
            'confusion_matrix': confmat_test.tolist(),
            # Для информации - храним оба набора
            'train_accuracy': float(accuracy_train),
            'train_precision': float(precision_train),
            'train_recall': float(recall_train),
            'train_f1': float(f1_train),
            'train_roc_auc': float(roc_auc_train),
        }

        print("\n" + "=" * 60)

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

    def classify_movie(self, movie_title):
        """
        Классификация фильма (успешный или нет)
        """
        if self.model is None:
            raise ValueError("Модель не обучена")

        # Ищем фильм в датасете
        movie = self.find_movie_by_title(movie_title)

        if movie is None:
            return None

        # Извлекаем признаки
        vote_average = float(movie['vote_average']) if pd.notna(movie['vote_average']) else 0
        vote_count = float(movie['vote_count']) if pd.notna(movie['vote_count']) else 0
        runtime = float(movie['runtime']) if pd.notna(movie['runtime']) else 0
        popularity = float(movie['popularity']) if pd.notna(movie['popularity']) else 0
        budget = float(movie['budget']) if pd.notna(movie['budget']) else 0
        main_genre = self._extract_main_genre(movie['genres'])
        original_language = str(movie['original_language']) if pd.notna(movie['original_language']) else 'en'

        # Получаем количество актеров
        cast_count = self._extract_cast_count(movie['id'])

        # Подготавливаем признаки
        features = pd.DataFrame({
            'vote_average': [vote_average],
            'vote_count': [vote_count],
            'runtime': [runtime],
            'popularity': [popularity],
            'cast_count': [cast_count],
            'budget': [budget],
            'log_vote_count': [np.log1p(vote_count)],
            'log_popularity': [np.log1p(popularity)],
            'log_budget': [np.log1p(budget)],
            'vote_per_cast': [vote_count / cast_count if cast_count > 0 else 0],
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

        # Масштабируем
        features_scaled = self.scaler.transform(features)

        # Предсказание
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]

        return {
            'title': movie['title'],
            'release_date': str(movie['release_date'])[:4] if pd.notna(movie['release_date']) else 'N/A',
            'vote_average': vote_average,
            'vote_count': vote_count,
            'popularity': popularity,
            'budget': budget,
            'runtime': runtime,
            'cast_count': cast_count,
            'genres': main_genre,
            'is_successful': int(prediction),
            'is_successful_text': '✅ Успешный фильм' if prediction == 1 else '❌ Менее успешный фильм',
            'probability_unsuccessful': float(probability[0]),
            'probability_successful': float(probability[1])
        }

    def export_tree_to_dot(self, filename='tree_classifier.dot'):
        """Экспорт дерева в формат GraphViz"""
        if self.model is None:
            raise ValueError("Модель не обучена")

        export_graphviz(
            self.model,
            out_file=filename,
            class_names=['Неуспешный', 'Успешный'],
            feature_names=self.feature_columns,
            impurity=False,
            filled=True
        )
        print(f"\n💾 Дерево экспортировано в {filename}")
        print(f"   Для преобразования в PNG используйте: https://onlineconvertfree.com/ru/convert/dot/")

    def save_model(self, filepath='movie_classifier_tree.pkl'):
        """Сохранение модели"""

        if self.model is None:
            raise ValueError("Модель не обучена")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'label_encoders': self.label_encoders,
            'metrics': self.metrics
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\n💾 Модель дерева классификации сохранена: {filepath}")

    def load_model(self, filepath='movie_classifier_tree.pkl'):
        """Загрузка модели"""

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.label_encoders = model_data['label_encoders']
        self.metrics = model_data.get('metrics', {})

        print(f"✅ Модель дерева классификации загружена: {filepath}")

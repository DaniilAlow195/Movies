import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
import pickle
import os
from ast import literal_eval

warnings.filterwarnings('ignore')


class MovieClassifier:
    """
    Классификатор фильмов для определения успешности/популярности фильма
    
    Классификация: 0 - Низкая успешность, 1 - Высокая успешность
    На основе: голосов, рейтинга, популярности, бюджета и других факторов
    """
    
    def __init__(self, movies_path, credits_path=None):
        """Инициализация классификатора"""
        self.movies_df = pd.read_csv(movies_path)
        self.credits_df = pd.read_csv(credits_path) if credits_path else None
        
        self.logistic_model = None
        self.knn_model = None
        self.scaler = None
        self.feature_columns = None
        self.label_encoders = {}
        self.prepared_data = None
        self.metrics_logistic = None
        self.metrics_knn = None
        self.threshold = 0.5
        
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
        print(f"   Распределение: {successful_count}/{len(df)} = {successful_count/len(df)*100:.1f}%")
        
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
    
    def train_models(self, test_size=0.2, random_state=42):
        """Обучение обеих моделей (Логистическая регрессия и kNN)"""
        print("\n" + "="*60)
        print("🎬 ОБУЧЕНИЕ КЛАССИФИКАТОРОВ ФИЛЬМОВ")
        print("="*60)
        
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
        
        # ====== ОБУЧЕНИЕ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ ======
        print("\n" + "-"*60)
        print("🔵 ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ")
        print("-"*60)
        
        self.logistic_model = LogisticRegression(fit_intercept=True, max_iter=1000)
        self.logistic_model.fit(X_train_scaled, y_train)
        
        # Предсказания
        y_pred_logistic_test = self.logistic_model.predict(X_test_scaled)
        y_pred_logistic_train = self.logistic_model.predict(X_train_scaled)
        
        # Метрики на тестовых данных
        accuracy_logistic = accuracy_score(y_test, y_pred_logistic_test)
        precision_logistic = precision_score(y_test, y_pred_logistic_test)
        recall_logistic = recall_score(y_test, y_pred_logistic_test)
        f1_logistic = f1_score(y_test, y_pred_logistic_test)
        roc_auc_logistic = roc_auc_score(y_test, y_pred_logistic_test)
        
        confmat_logistic = confusion_matrix(y_test, y_pred_logistic_test)
        
        print(f"\n📈 Метрики на тестовых данных:")
        print(f"   Accuracy (правильность): {accuracy_logistic:.3f}")
        print(f"   Precision (точность): {precision_logistic:.3f}")
        print(f"   Recall (полнота): {recall_logistic:.3f}")
        print(f"   F1-score: {f1_logistic:.3f}")
        print(f"   ROC AUC: {roc_auc_logistic:.3f}")
        print(f"\n   Матрица ошибок:")
        print(f"   [[TN={confmat_logistic[0,0]} FP={confmat_logistic[0,1]}]")
        print(f"    [FN={confmat_logistic[1,0]} TP={confmat_logistic[1,1]}]]")
        
        # Важность признаков (коэффициенты)
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'coefficient': self.logistic_model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        
        print(f"\n⭐ Топ-5 значимых признаков:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"   {row['feature']}: {row['coefficient']:.4f}")
        
        self.metrics_logistic = {
            'accuracy': float(accuracy_logistic),
            'precision': float(precision_logistic),
            'recall': float(recall_logistic),
            'f1': float(f1_logistic),
            'roc_auc': float(roc_auc_logistic),
            'confusion_matrix': confmat_logistic.tolist()
        }
        
        # ====== ОБУЧЕНИЕ k-NN ======
        print("\n" + "-"*60)
        print("🟢 МЕТОД k-БЛИЖАЙШИХ СОСЕДЕЙ (k=5)")
        print("-"*60)
        
        self.knn_model = KNeighborsClassifier(n_neighbors=5)
        self.knn_model.fit(X_train_scaled, y_train)
        
        # Предсказания
        y_pred_knn_test = self.knn_model.predict(X_test_scaled)
        y_pred_knn_train = self.knn_model.predict(X_train_scaled)
        
        # Метрики на тестовых данных
        accuracy_knn = accuracy_score(y_test, y_pred_knn_test)
        precision_knn = precision_score(y_test, y_pred_knn_test)
        recall_knn = recall_score(y_test, y_pred_knn_test)
        f1_knn = f1_score(y_test, y_pred_knn_test)
        roc_auc_knn = roc_auc_score(y_test, y_pred_knn_test)
        
        confmat_knn = confusion_matrix(y_test, y_pred_knn_test)
        
        print(f"\n📈 Метрики на тестовых данных:")
        print(f"   Accuracy (правильность): {accuracy_knn:.3f}")
        print(f"   Precision (точность): {precision_knn:.3f}")
        print(f"   Recall (полнота): {recall_knn:.3f}")
        print(f"   F1-score: {f1_knn:.3f}")
        print(f"   ROC AUC: {roc_auc_knn:.3f}")
        print(f"\n   Матрица ошибок:")
        print(f"   [[TN={confmat_knn[0,0]} FP={confmat_knn[0,1]}]")
        print(f"    [FN={confmat_knn[1,0]} TP={confmat_knn[1,1]}]]")
        
        self.metrics_knn = {
            'accuracy': float(accuracy_knn),
            'precision': float(precision_knn),
            'recall': float(recall_knn),
            'f1': float(f1_knn),
            'roc_auc': float(roc_auc_knn),
            'confusion_matrix': confmat_knn.tolist()
        }
        
        # СРАВНЕНИЕ МОДЕЛЕЙ
        print("\n" + "="*60)
        print("📊 СРАВНЕНИЕ МОДЕЛЕЙ")
        print("="*60)
        print(f"\n              Logistic Reg.  |  k-NN (k=5)")
        print(f"   Accuracy:     {accuracy_logistic:.3f}        |     {accuracy_knn:.3f}")
        print(f"   Precision:    {precision_logistic:.3f}        |     {precision_knn:.3f}")
        print(f"   Recall:       {recall_logistic:.3f}        |     {recall_knn:.3f}")
        print(f"   F1-score:     {f1_logistic:.3f}        |     {f1_knn:.3f}")
        print(f"   ROC AUC:      {roc_auc_logistic:.3f}        |     {roc_auc_knn:.3f}")
        
        # Выбираем лучшую модель
        best_model = 'k-NN' if accuracy_knn > accuracy_logistic else 'Logistic Regression'
        print(f"\n✨ Лучшая модель по Accuracy: {best_model}")
        
        return {
            'logistic': self.metrics_logistic,
            'knn': self.metrics_knn
        }
    
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
    
    def classify_movie(self, movie_title, use_model='knn'):
        """
        Классификация фильма (успешный или нет)
        use_model: 'logistic' или 'knn'
        """
        if self.logistic_model is None or self.knn_model is None:
            raise ValueError("Модели не обучены")
        
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
        
        # Выбираем модель
        if use_model == 'knn':
            prediction = self.knn_model.predict(features_scaled)[0]
            probability = None  # kNN не дает вероятностей
        else:  # logistic
            prediction = self.logistic_model.predict(features_scaled)[0]
            probability = self.logistic_model.predict_proba(features_scaled)[0]
        
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
            'probability_unsuccessful': float(probability[0]) if probability is not None else None,
            'probability_successful': float(probability[1]) if probability is not None else None,
            'model_used': use_model
        }
    
    def save_models(self, filepath_logistic='logistic_model.pkl', filepath_knn='knn_model.pkl'):
        """Сохранение обеих моделей"""
        
        if self.logistic_model is None or self.knn_model is None:
            raise ValueError("Модели не обучены")
        
        # Сохраняем логистическую регрессию
        logistic_data = {
            'model': self.logistic_model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'label_encoders': self.label_encoders,
            'metrics': self.metrics_logistic
        }
        
        with open(filepath_logistic, 'wb') as f:
            pickle.dump(logistic_data, f)
        
        print(f"\n💾 Модель логистической регрессии сохранена: {filepath_logistic}")
        
        # Сохраняем k-NN
        knn_data = {
            'model': self.knn_model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'label_encoders': self.label_encoders,
            'metrics': self.metrics_knn
        }
        
        with open(filepath_knn, 'wb') as f:
            pickle.dump(knn_data, f)
        
        print(f"💾 Модель k-NN сохранена: {filepath_knn}")
    
    def load_models(self, filepath_logistic='logistic_model.pkl', filepath_knn='knn_model.pkl'):
        """Загрузка обеих моделей"""
        
        # Загружаем логистическую регрессию
        with open(filepath_logistic, 'rb') as f:
            logistic_data = pickle.load(f)
        
        self.logistic_model = logistic_data['model']
        self.scaler = logistic_data['scaler']
        self.feature_columns = logistic_data['feature_columns']
        self.label_encoders = logistic_data['label_encoders']
        self.metrics_logistic = logistic_data.get('metrics', {})
        
        print(f"✅ Модель логистической регрессии загружена: {filepath_logistic}")
        
        # Загружаем k-NN
        with open(filepath_knn, 'rb') as f:
            knn_data = pickle.load(f)
        
        self.knn_model = knn_data['model']
        self.metrics_knn = knn_data.get('metrics', {})
        
        print(f"✅ Модель k-NN загружена: {filepath_knn}")

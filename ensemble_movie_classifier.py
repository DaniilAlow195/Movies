import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.svm import SVC
import warnings
import pickle
import os
from ast import literal_eval
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class EnsembleMovieClassifier:
    """
    Классификатор фильмов на основе ансамблевых методов (Random Forest и Gradient Boosting)
    для определения успешности/популярности фильма.
    
    Классификация: 0 - Низкая успешность, 1 - Высокая успешность
    На основе: голосов, рейтинга, популярности, бюджета и других факторов
    """
    
    def __init__(self, movies_path, credits_path=None):
        """Инициализация классификатора"""
        self.movies_df = pd.read_csv(movies_path)
        self.credits_df = pd.read_csv(credits_path) if credits_path else None
        
        self.random_forest_model = None
        self.gradient_boosting_model = None
        self.logistic_model = None
        self.knn_model = None
        self.svm_model = None
        self.decision_tree_model = None
        
        self.scaler = None
        self.feature_columns = None
        self.label_encoders = {}
        self.prepared_data = None
        
        self.metrics_rf = None
        self.metrics_gb = None
        self.metrics_logistic = None
        self.metrics_knn = None
        self.metrics_svm = None
        self.metrics_tree = None
        self.comparison_results = None
        
        print(f"\n📊 Диагностика данных для ансамблевого классификатора:")
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
    
    def train_ensemble_models(self, test_size=0.2, random_state=0):
        """Обучение ансамблевых моделей (Random Forest и Gradient Boosting)"""
        print("\n" + "="*70)
        print("🎬 КЛАССИФИКАЦИЯ ФИЛЬМОВ НА ОСНОВЕ АНСАМБЛЕВЫХ МЕТОДОВ")
        print("="*70)
        
        print("\n📊 Подготовка данных...")
        features, target = self._prepare_data()
        
        self.feature_columns = features.columns.tolist()
        
        print(f"✅ Используется {len(self.feature_columns)} признаков")
        print(f"   Признаки: {self.feature_columns}")
        
        # Разделяем данные на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state
        )
        
        print(f"\n📋 Размер обучающего массива факторных признаков: {X_train.shape}")
        print(f"   Размер тестового массива факторных признаков: {X_test.shape}")
        print(f"   Размер обучающего массива результативного признака: {y_train.shape}")
        print(f"   Размер тестового массива результативного признака: {y_test.shape}")
        
        # Масштабируем данные
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # ====== RANDOM FOREST ======
        print("\n" + "-"*70)
        print("🌲 МЕТОД: RANDOM FOREST (СЛУЧАЙНЫЙ ЛЕС)")
        print("-"*70)
        
        self.random_forest_model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            max_depth=3,
            max_features=3
        )
        self.random_forest_model.fit(X_train, y_train)
        
        # Предсказания на обучающих данных
        y_pred_rf_train = self.random_forest_model.predict(X_train)
        confmat_rf_train = confusion_matrix(y_train, y_pred_rf_train)
        
        print(f"\n📈 Метрики на обучающих данных:")
        print(f"   Матрица ошибок:")
        print(f"   TN = {confmat_rf_train[0,0]}   FP = {confmat_rf_train[0,1]}")
        print(f"   FN = {confmat_rf_train[1,0]}   TP = {confmat_rf_train[1,1]}")
        
        accuracy_rf_train = accuracy_score(y_train, y_pred_rf_train)
        precision_rf_train = precision_score(y_train, y_pred_rf_train)
        recall_rf_train = recall_score(y_train, y_pred_rf_train)
        f1_rf_train = f1_score(y_train, y_pred_rf_train)
        roc_auc_rf_train = roc_auc_score(y_train, y_pred_rf_train)
        
        print(f"\n   Accuracy (правильность): {accuracy_rf_train:.3f}")
        print(f"   Precision (точность): {precision_rf_train:.3f}")
        print(f"   Recall (полнота): {recall_rf_train:.3f}")
        print(f"   F1: {f1_rf_train:.3f}")
        print(f"   ROC AUC: {roc_auc_rf_train:.3f}")
        
        # Предсказания на тестовых данных
        y_pred_rf_test = self.random_forest_model.predict(X_test)
        confmat_rf_test = confusion_matrix(y_test, y_pred_rf_test)
        
        print(f"\n📈 Метрики на тестовых данных:")
        print(f"   Матрица ошибок:")
        print(f"   TN = {confmat_rf_test[0,0]}   FP = {confmat_rf_test[0,1]}")
        print(f"   FN = {confmat_rf_test[1,0]}   TP = {confmat_rf_test[1,1]}")
        
        accuracy_rf = accuracy_score(y_test, y_pred_rf_test)
        precision_rf = precision_score(y_test, y_pred_rf_test)
        recall_rf = recall_score(y_test, y_pred_rf_test)
        f1_rf = f1_score(y_test, y_pred_rf_test)
        roc_auc_rf = roc_auc_score(y_test, y_pred_rf_test)
        
        print(f"\n   Accuracy (правильность): {accuracy_rf:.3f}")
        print(f"   Precision (точность): {precision_rf:.3f}")
        print(f"   Recall (полнота): {recall_rf:.3f}")
        print(f"   F1: {f1_rf:.3f}")
        print(f"   ROC AUC: {roc_auc_rf:.3f}")
        
        # Важность признаков
        importances_rf = self.random_forest_model.feature_importances_
        indices_rf = np.argsort(importances_rf)[::-1]
        
        print(f"\n⭐ Важность признаков для Random Forest:")
        for f in range(len(self.feature_columns)):
            print(f"   {f + 1}) {self.feature_columns[indices_rf[f]]:<20} {importances_rf[indices_rf[f]]:.4f}")
        
        self.metrics_rf = {
            'accuracy': float(accuracy_rf),
            'precision': float(precision_rf),
            'recall': float(recall_rf),
            'f1': float(f1_rf),
            'roc_auc': float(roc_auc_rf),
            'confusion_matrix': confmat_rf_test.tolist()
        }
        
        # ====== GRADIENT BOOSTING ======
        print("\n" + "-"*70)
        print("⬆️  МЕТОД: GRADIENT BOOSTING (ГРАДИЕНТНЫЙ БУСТИНГ)")
        print("-"*70)
        
        self.gradient_boosting_model = GradientBoostingClassifier(
            random_state=random_state,
            n_estimators=100,
            max_depth=3
        )
        self.gradient_boosting_model.fit(X_train, y_train)
        
        # Предсказания на обучающих данных
        y_pred_gb_train = self.gradient_boosting_model.predict(X_train)
        confmat_gb_train = confusion_matrix(y_train, y_pred_gb_train)
        
        print(f"\n📈 Метрики на обучающих данных:")
        print(f"   Матрица ошибок:")
        print(f"   TN = {confmat_gb_train[0,0]}   FP = {confmat_gb_train[0,1]}")
        print(f"   FN = {confmat_gb_train[1,0]}   TP = {confmat_gb_train[1,1]}")
        
        accuracy_gb_train = accuracy_score(y_train, y_pred_gb_train)
        precision_gb_train = precision_score(y_train, y_pred_gb_train)
        recall_gb_train = recall_score(y_train, y_pred_gb_train)
        f1_gb_train = f1_score(y_train, y_pred_gb_train)
        roc_auc_gb_train = roc_auc_score(y_train, y_pred_gb_train)
        
        print(f"\n   Accuracy (правильность): {accuracy_gb_train:.3f}")
        print(f"   Precision (точность): {precision_gb_train:.3f}")
        print(f"   Recall (полнота): {recall_gb_train:.3f}")
        print(f"   F1: {f1_gb_train:.3f}")
        print(f"   ROC AUC: {roc_auc_gb_train:.3f}")
        
        # Предсказания на тестовых данных
        y_pred_gb_test = self.gradient_boosting_model.predict(X_test)
        confmat_gb_test = confusion_matrix(y_test, y_pred_gb_test)
        
        print(f"\n📈 Метрики на тестовых данных:")
        print(f"   Матрица ошибок:")
        print(f"   TN = {confmat_gb_test[0,0]}   FP = {confmat_gb_test[0,1]}")
        print(f"   FN = {confmat_gb_test[1,0]}   TP = {confmat_gb_test[1,1]}")
        
        accuracy_gb = accuracy_score(y_test, y_pred_gb_test)
        precision_gb = precision_score(y_test, y_pred_gb_test)
        recall_gb = recall_score(y_test, y_pred_gb_test)
        f1_gb = f1_score(y_test, y_pred_gb_test)
        roc_auc_gb = roc_auc_score(y_test, y_pred_gb_test)
        
        print(f"\n   Accuracy (правильность): {accuracy_gb:.3f}")
        print(f"   Precision (точность): {precision_gb:.3f}")
        print(f"   Recall (полнота): {recall_gb:.3f}")
        print(f"   F1: {f1_gb:.3f}")
        print(f"   ROC AUC: {roc_auc_gb:.3f}")
        
        # Важность признаков
        importances_gb = self.gradient_boosting_model.feature_importances_
        indices_gb = np.argsort(importances_gb)[::-1]
        
        print(f"\n⭐ Важность признаков для Gradient Boosting:")
        for f in range(len(self.feature_columns)):
            print(f"   {f + 1}) {self.feature_columns[indices_gb[f]]:<20} {importances_gb[indices_gb[f]]:.4f}")
        
        self.metrics_gb = {
            'accuracy': float(accuracy_gb),
            'precision': float(precision_gb),
            'recall': float(recall_gb),
            'f1': float(f1_gb),
            'roc_auc': float(roc_auc_gb),
            'confusion_matrix': confmat_gb_test.tolist()
        }
        
        # ====== СРАВНЕНИЕ МОДЕЛЕЙ ======
        return self._compare_all_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
    
    def _compare_all_models(self, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
        """Сравнение всех моделей"""
        print("\n" + "="*70)
        print("📊 СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ")
        print("="*70)
        
        # Логистическая регрессия
        print("\n🔵 Обучение: Логистическая регрессия")
        self.logistic_model = LogisticRegression(fit_intercept=True, max_iter=1000)
        self.logistic_model.fit(X_train_scaled, y_train)
        y_pred_logistic = self.logistic_model.predict(X_test_scaled)
        accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
        self.metrics_logistic = {'accuracy': float(accuracy_logistic)}
        print(f"   Accuracy: {accuracy_logistic:.3f}")
        
        # Метод опорных векторов (SVM)
        print("🟠 Обучение: Метод опорных векторов (SVM)")
        self.svm_model = SVC(kernel='linear', class_weight='balanced')
        self.svm_model.fit(X_train_scaled, y_train)
        y_pred_svm = self.svm_model.predict(X_test_scaled)
        accuracy_svm = accuracy_score(y_test, y_pred_svm)
        self.metrics_svm = {'accuracy': float(accuracy_svm)}
        print(f"   Accuracy: {accuracy_svm:.3f}")
        
        # k-NN
        print("🟢 Обучение: k-NN")
        self.knn_model = KNeighborsClassifier(n_neighbors=5)
        self.knn_model.fit(X_train_scaled, y_train)
        y_pred_knn = self.knn_model.predict(X_test_scaled)
        accuracy_knn = accuracy_score(y_test, y_pred_knn)
        self.metrics_knn = {'accuracy': float(accuracy_knn)}
        print(f"   Accuracy: {accuracy_knn:.3f}")
        
        # Дерево классификации
        print("🟡 Обучение: Дерево классификации")
        self.decision_tree_model = DecisionTreeClassifier(max_depth=3)
        self.decision_tree_model.fit(X_train, y_train)
        y_pred_tree = self.decision_tree_model.predict(X_test)
        accuracy_tree = accuracy_score(y_test, y_pred_tree)
        self.metrics_tree = {'accuracy': float(accuracy_tree)}
        print(f"   Accuracy: {accuracy_tree:.3f}")
        
        # Таблица сравнения
        print("\n" + "-"*70)
        print("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
        print("-"*70)
        print(f"\n{'Модель':<25} | {'Accuracy':<10}")
        print("-" * 40)
        print(f"{'Логистическая регрессия':<25} | {accuracy_logistic:>8.3f}")
        print(f"{'Метод опорных векторов':<25} | {accuracy_svm:>8.3f}")
        print(f"{'k-NN (k=5)':<25} | {accuracy_knn:>8.3f}")
        print(f"{'Дерево классификации':<25} | {accuracy_tree:>8.3f}")
        print(f"{'Random Forest':<25} | {self.metrics_rf[\"accuracy\"]:>8.3f}")
        print(f"{'Gradient Boosting':<25} | {self.metrics_gb[\"accuracy\"]:>8.3f}")
        
        # Определяем лучшую модель
        accuracies = {
            'Логистическая регрессия': accuracy_logistic,
            'SVM': accuracy_svm,
            'k-NN': accuracy_knn,
            'Дерево классификации': accuracy_tree,
            'Random Forest': self.metrics_rf['accuracy'],
            'Gradient Boosting': self.metrics_gb['accuracy']
        }
        
        best_model = max(accuracies, key=accuracies.get)
        best_accuracy = accuracies[best_model]
        
        print(f"\n✨ Лучшая модель: {best_model} (Accuracy = {best_accuracy:.3f})")
        
        self.comparison_results = accuracies
        
        return {
            'random_forest': self.metrics_rf,
            'gradient_boosting': self.metrics_gb,
            'logistic': self.metrics_logistic,
            'svm': self.metrics_svm,
            'knn': self.metrics_knn,
            'decision_tree': self.metrics_tree,
            'comparison': accuracies,
            'best_model': best_model
        }
    
    def get_feature_importances(self):
        """Получить важность признаков для ансамблевых моделей"""
        print("\n" + "="*70)
        print("⭐ СРАВНЕНИЕ ВАЖНОСТИ ПРИЗНАКОВ")
        print("="*70)
        
        if self.random_forest_model is None or self.gradient_boosting_model is None:
            raise ValueError("Модели не обучены")
        
        importances_rf = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.random_forest_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importances_gb = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.gradient_boosting_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🌲 Random Forest - Топ-10 признаков:")
        print(importances_rf.head(10).to_string(index=False))
        
        print("\n⬆️  Gradient Boosting - Топ-10 признаков:")
        print(importances_gb.head(10).to_string(index=False))
        
        return {
            'random_forest': importances_rf,
            'gradient_boosting': importances_gb
        }
    
    def save_ensemble_models(self, filepath_rf='random_forest_model.pkl', 
                            filepath_gb='gradient_boosting_model.pkl'):
        """Сохранение ансамблевых моделей"""
        
        if self.random_forest_model is None or self.gradient_boosting_model is None:
            raise ValueError("Модели не обучены")
        
        # Сохраняем Random Forest
        rf_data = {
            'model': self.random_forest_model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'label_encoders': self.label_encoders,
            'metrics': self.metrics_rf
        }
        
        with open(filepath_rf, 'wb') as f:
            pickle.dump(rf_data, f)
        
        print(f"\n💾 Модель Random Forest сохранена: {filepath_rf}")
        
        # Сохраняем Gradient Boosting
        gb_data = {
            'model': self.gradient_boosting_model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'label_encoders': self.label_encoders,
            'metrics': self.metrics_gb
        }
        
        with open(filepath_gb, 'wb') as f:
            pickle.dump(gb_data, f)
        
        print(f"💾 Модель Gradient Boosting сохранена: {filepath_gb}")
    
    def load_ensemble_models(self, filepath_rf='random_forest_model.pkl', 
                            filepath_gb='gradient_boosting_model.pkl'):
        """Загрузка ансамблевых моделей"""
        
        # Загружаем Random Forest
        with open(filepath_rf, 'rb') as f:
            rf_data = pickle.load(f)
        
        self.random_forest_model = rf_data['model']
        self.scaler = rf_data['scaler']
        self.feature_columns = rf_data['feature_columns']
        self.label_encoders = rf_data['label_encoders']
        self.metrics_rf = rf_data.get('metrics', {})
        
        print(f"✅ Модель Random Forest загружена: {filepath_rf}")
        
        # Загружаем Gradient Boosting
        with open(filepath_gb, 'rb') as f:
            gb_data = pickle.load(f)
        
        self.gradient_boosting_model = gb_data['model']
        self.metrics_gb = gb_data.get('metrics', {})
        
        print(f"✅ Модель Gradient Boosting загружена: {filepath_gb}")


# ============ ПРИМЕР ИСПОЛЬЗОВАНИЯ ============
if __name__ == "__main__":
    # Путь к данным
    movies_path = 'movies_metadata.csv'
    credits_path = 'credits.csv'
    
    # Инициализируем классификатор
    classifier = EnsembleMovieClassifier(movies_path, credits_path)
    
    # Обучаем ансамблевые модели и сравниваем все методы
    results = classifier.train_ensemble_models(test_size=0.2, random_state=0)
    
    # Получаем важность признаков
    importances = classifier.get_feature_importances()
    
    # Сохраняем модели
    classifier.save_ensemble_models('random_forest_model.pkl', 'gradient_boosting_model.pkl')

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from ast import literal_eval
import warnings
import os
import traceback

from revenue_predictor import RevenuePredictor
from movie_classifier import MovieClassifier
from ensemble_movie_classifier import EnsembleMovieClassifier

from revenue_predictor_tree import RevenuePredictorTree
from movie_classifier_tree import MovieClassifierTree

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)


class MovieRecommendationSystem:
    """Система рекомендации фильмов на основе TF-IDF и косинусного подобия"""

    def __init__(self, movies_path, keywords_path=None):
        """Инициализация системы рекомендаций фильмов"""
        print("📂 Загрузка данных фильмов...")
        self.movies = pd.read_csv(movies_path)
        print(f"✅ Загружено фильмов: {len(self.movies)}")

        self.keywords = pd.read_csv(keywords_path) if keywords_path else None
        if self.keywords is not None:
            print(f"✅ Загружено ключевых слов: {len(self.keywords)}")

        self.tfidf_matrix = None
        self._prepare_data()

    def _prepare_data(self):
        """Подготовка данных для анализа"""
        print("🔧 Подготовка данных...")

        self.movies['genres'] = self.movies['genres'].fillna('[]')
        self.movies['overview'] = self.movies['overview'].fillna('')
        self.movies['title'] = self.movies['title'].fillna('')
        self.movies['release_date'] = self.movies['release_date'].fillna('')

        self.movies['vote_average'] = pd.to_numeric(self.movies['vote_average'], errors='coerce').fillna(0)
        self.movies['popularity'] = pd.to_numeric(self.movies['popularity'], errors='coerce').fillna(0)

        self.movies['genres_text'] = self.movies['genres'].apply(self._extract_genres)

        if self.keywords is not None:
            try:
                if 'id' in self.keywords.columns and 'keywords' in self.keywords.columns:
                    self.movies = self.movies.merge(
                        self.keywords[['id', 'keywords']],
                        on='id',
                        how='left'
                    )
                    self.movies['keywords'] = self.movies['keywords'].fillna('[]')
                    self.movies['keywords_text'] = self.movies['keywords'].apply(self._extract_keywords)
                else:
                    self.movies['keywords_text'] = ''
            except Exception as e:
                print(f"⚠️ Ошибка при обработке ключевых слов: {e}")
                self.movies['keywords_text'] = ''
        else:
            self.movies['keywords_text'] = ''

        self.movies['text_features'] = (
                self.movies['genres_text'].fillna('') + ' ' +
                self.movies['overview'].fillna('') + ' ' +
                self.movies['keywords_text'].fillna('')
        )

        # Удаляем пустые строки
        self.movies = self.movies[self.movies['text_features'].str.len() > 0].reset_index(drop=True)
        print(f"✅ Фильмов для обработки TF-IDF: {len(self.movies)}")

        # Вычисляем TF-IDF матрицу
        print("⏳ Вычисление TF-IDF матрицы...")
        tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            min_df=1,
            max_df=0.95
        )
        self.tfidf_matrix = tfidf.fit_transform(self.movies['text_features'])
        print(f"✅ TF-IDF матрица создана (разреженная матрица, размер в памяти: оптимален)")
        print("✅ Данные готовы!\n")

    def _extract_genres(self, genres_str):
        """Извлечение названий жанров из JSON строки"""
        try:
            if isinstance(genres_str, str) and genres_str != '[]':
                genres_list = literal_eval(genres_str)
                if isinstance(genres_list, list):
                    return ' '.join([str(g.get('name', '')) for g in genres_list if g.get('name')])
        except Exception:
            pass
        return ''

    def _extract_keywords(self, keywords_str):
        """Извлечение ключевых слов из JSON строки"""
        try:
            if isinstance(keywords_str, str) and keywords_str != '[]':
                keywords_list = literal_eval(keywords_str)
                if isinstance(keywords_list, list):
                    return ' '.join([str(k.get('name', '')) for k in keywords_list if k.get('name')])
        except Exception:
            pass
        return ''

    def find_movie(self, title):
        """Поиск фильма по названию"""
        if not title:
            return None

        title_lower = title.lower().strip()

        # Точный поиск
        exact_matches = self.movies[self.movies['title'].str.lower() == title_lower]
        if len(exact_matches) > 0:
            return exact_matches.index[0]

        # Поиск по частичному совпадению
        partial_matches = self.movies[
            self.movies['title'].str.lower().str.contains(title_lower, na=False, regex=False)
        ]

        if len(partial_matches) > 0:
            return partial_matches.index[0]

        return None

    def get_recommendations(self, movie_title, n_recommendations=10):
        """Получить рекомендации похожих фильмов"""
        idx = self.find_movie(movie_title)

        if idx is None:
            return None

        # Используем разреженную матрицу для расчета подобия ТОЛЬКО для одного фильма
        print(f"   🔍 Вычисление подобия для фильма #{idx}...")

        movie_vector = self.tfidf_matrix[idx]

        # Вычисляем подобие только для данного фильма со всеми остальными
        # Это намного быстрее и экономнее по памяти!
        sim_scores = cosine_similarity(movie_vector, self.tfidf_matrix)[0]

        # Создаем список с индексами и оценками
        sim_scores = list(enumerate(sim_scores))

        # Сортируем по оценке подобия
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Получаем индексы n наиболее похожих фильмов (исключая сам фильм)
        sim_scores = sim_scores[1:n_recommendations + 1]

        if len(sim_scores) == 0:
            return None

        movie_indices = [i[0] for i in sim_scores]

        # Получаем информацию о рекомендованных фильмах
        recommendations = self.movies.iloc[movie_indices][[
            'title', 'release_date', 'vote_average', 'popularity', 'genres_text', 'overview'
        ]].copy()

        # Добавляем оценку подобия
        recommendations['similarity_score'] = [float(score[1]) for score in sim_scores]
        recommendations = recommendations.reset_index(drop=True)

        return recommendations

    def get_search_suggestions(self, query):
        """Получить подсказки для поиска"""
        if not query or len(query) < 1:
            return []

        query_lower = query.lower().strip()

        try:
            # Ищем фильмы, содержащие запрос
            matching_movies = self.movies[
                self.movies['title'].str.lower().str.contains(query_lower, na=False, regex=False)
            ]

            # Возвращаем уникальные названия (максимум 10)
            suggestions = matching_movies['title'].unique()[:10].tolist()

            return suggestions
        except Exception as e:
            print(f"❌ Ошибка в get_search_suggestions: {e}")
            traceback.print_exc()
            return []


# ============ ИНИЦИАЛИЗАЦИЯ ============

print("=" * 70)
print("🎬 MOVIEMATCH - СИСТЕМА РЕКОМЕНДАЦИЙ И АНАЛИЗА ФИЛЬМОВ")
print("=" * 70 + "\n")

# Пути к файлам - ИЗМЕНИТЕ НА СВОИ ПУТИ!
movies_path = r"C:\Users\Даниил\Downloads\archive\movies_metadata.csv"
keywords_path = r"C:\Users\Даниил\Downloads\archive\keywords.csv"
credits_path = r"C:\Users\Даниил\Downloads\archive\credits.csv"

# Проверяем существование файлов
print("🔍 Проверка файлов данных:")
for path, name in [(movies_path, "movies_metadata.csv"),
                   (keywords_path, "keywords.csv"),
                   (credits_path, "credits.csv")]:
    if os.path.exists(path):
        print(f"  ✅ {name} найден")
    else:
        print(f"  ❌ {name} НЕ НАЙДЕН: {path}")

print()

# ============ ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ РЕКОМЕНДАЦИЙ ============

system = None
try:
    system = MovieRecommendationSystem(movies_path, keywords_path)
    print("✅ Система рекомендаций загружена успешно!\n")
except Exception as e:
    print(f"❌ ОШИБКА при загрузке системы: {e}")
    traceback.print_exc()
    print()

# ============ ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ РЕГРЕССИИ (Прогноз доходов) ============

# Предсказатель доходов (оригинальная модель - Gradient Boosting)
revenue_predictor = None
model_file = 'revenue_model.pkl'

# Предсказатель доходов (новая модель - Decision Tree)
revenue_predictor_tree = None
model_tree_file = 'revenue_model_tree.pkl'

print("=" * 70)
print("💰 ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ ПРОГНОЗА ДОХОДОВ")
print("=" * 70 + "\n")

# Gradient Boosting модель
print("📌 Модель 1: Gradient Boosting Regressor (оригинальная)")
print("-" * 70)
try:
    if os.path.exists(model_file):
        print(f"📂 Найдена сохраненная модель: {model_file}")
        revenue_predictor = RevenuePredictor(movies_path, credits_path)
        revenue_predictor.load_model(model_file)
    else:
        print(f"🔨 Обученная модель не найдена. Обучаем новую...\n")
        revenue_predictor = RevenuePredictor(movies_path, credits_path)
        metrics = revenue_predictor.train_model()
        revenue_predictor.save_model(model_file)

    print("✅ Gradient Boosting модель инициализирована!\n")
except Exception as e:
    print(f"❌ ОШИБКА при инициализации Gradient Boosting: {e}")
    traceback.print_exc()
    revenue_predictor = None
    print()

# Decision Tree модель
print("📌 Модель 2: Decision Tree Regressor (новая)")
print("-" * 70)
try:
    if os.path.exists(model_tree_file):
        print(f"📂 Найдена сохраненная модель: {model_tree_file}")
        revenue_predictor_tree = RevenuePredictorTree(movies_path, credits_path)
        revenue_predictor_tree.load_model(model_tree_file)
    else:
        print(f"🔨 Обученная модель не найдена. Обучаем новую...\n")
        revenue_predictor_tree = RevenuePredictorTree(movies_path, credits_path)
        metrics = revenue_predictor_tree.train_model()
        revenue_predictor_tree.save_model(model_tree_file)

    print("✅ Decision Tree модель инициализирована!\n")
except Exception as e:
    print(f"❌ ОШИБКА при инициализации Decision Tree: {e}")
    traceback.print_exc()
    revenue_predictor_tree = None
    print()

# ============ ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ КЛАССИФИКАЦИИ (Успешность фильма) ============

# Классификатор фильмов (оригинальная модель - Logistic Regression и k-NN)
movie_classifier = None
logistic_model_file = 'logistic_model.pkl'
knn_model_file = 'knn_model.pkl'

# Классификатор фильмов (новая модель - Decision Tree)
movie_classifier_tree = None
tree_classifier_file = 'movie_classifier_tree.pkl'

# Ансамблевый классификатор фильмов (Random Forest и Gradient Boosting)
ensemble_classifier = None
ensemble_rf_file = 'random_forest_model.pkl'
ensemble_gb_file = 'gradient_boosting_model.pkl'

print("=" * 70)
print("🎬 ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ КЛАССИФИКАЦИИ")
print("=" * 70 + "\n")

# Logistic Regression и k-NN модели
print("📌 Модель 1: Логистическая регрессия + k-NN (оригинальные)")
print("-" * 70)
try:
    if os.path.exists(logistic_model_file) and os.path.exists(knn_model_file):
        print(f"📂 Найдены сохраненные модели")
        movie_classifier = MovieClassifier(movies_path, credits_path)
        movie_classifier.load_models(logistic_model_file, knn_model_file)
    else:
        print(f"🔨 Обученные модели не найдены. Обучаем новые...\n")
        movie_classifier = MovieClassifier(movies_path, credits_path)
        metrics = movie_classifier.train_models()
        movie_classifier.save_models(logistic_model_file, knn_model_file)

    print("✅ Логистическая регрессия + k-NN инициализированы!\n")
except Exception as e:
    print(f"❌ ОШИБКА при инициализации LR+kNN: {e}")
    traceback.print_exc()
    movie_classifier = None
    print()

# Decision Tree классификатор
print("📌 Модель 2: Decision Tree Classifier (новая)")
print("-" * 70)
try:
    if os.path.exists(tree_classifier_file):
        print(f"📂 Найдена сохраненная модель: {tree_classifier_file}")
        movie_classifier_tree = MovieClassifierTree(movies_path, credits_path)
        movie_classifier_tree.load_model(tree_classifier_file)
    else:
        print(f"🔨 Обученная модель не найдена. Обучаем новую...\n")
        movie_classifier_tree = MovieClassifierTree(movies_path, credits_path)
        metrics = movie_classifier_tree.train_model()
        movie_classifier_tree.save_model(tree_classifier_file)

    print("✅ Decision Tree модель инициализирована!\n")
except Exception as e:
    print(f"❌ ОШИБКА при инициализации Decision Tree: {e}")
    traceback.print_exc()
    movie_classifier_tree = None
    print()

# Ансамблевый классификатор (Random Forest и Gradient Boosting)
print("📌 Модель 3: Ансамблевые методы (Random Forest + Gradient Boosting)")
print("-" * 70)
try:
    if os.path.exists(ensemble_rf_file) and os.path.exists(ensemble_gb_file):
        print(f"📂 Найдены сохраненные модели")
        ensemble_classifier = EnsembleMovieClassifier(movies_path, credits_path)
        ensemble_classifier.load_ensemble_models(ensemble_rf_file, ensemble_gb_file)
    else:
        print(f"🔨 Обученные модели не найдены. Обучаем новые...\n")
        ensemble_classifier = EnsembleMovieClassifier(movies_path, credits_path)
        metrics = ensemble_classifier.train_ensemble_models()
        ensemble_classifier.save_ensemble_models(ensemble_rf_file, ensemble_gb_file)

    print("✅ Ансамблевые модели инициализированы!\n")
except Exception as e:
    print(f"❌ ОШИБКА при инициализации ансамблевых моделей: {e}")
    traceback.print_exc()
    ensemble_classifier = None
    print()

print("=" * 70)
print("🚀 ПРИЛОЖЕНИЕ ГОТОВО К РАБОТЕ")
print("=" * 70)
print("\n📍 Откройте браузер: http://127.0.0.1:5000\n")


# ============ МАРШРУТЫ API ============

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')


@app.route('/api/search', methods=['GET'])
def search_suggestions():
    """API для поиска подсказок при вводе названия фильма"""
    try:
        query = request.args.get('q', '').strip()

        if not system:
            print("❌ Система не загружена!")
            return jsonify({
                'error': 'System not loaded',
                'suggestions': []
            }), 500

        if len(query) < 1:
            return jsonify({'suggestions': []})

        print(f"🔍 Поиск подсказок: '{query}'")
        suggestions = system.get_search_suggestions(query)
        print(f"   → найдено {len(suggestions)} подсказок")

        return jsonify({'suggestions': suggestions})

    except Exception as e:
        print(f"❌ ОШИБКА в /api/search: {e}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'suggestions': []
        }), 500


@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """API для получения рекомендаций похожих фильмов"""
    try:
        data = request.json
        movie_title = data.get('movie_title', '').strip()
        n_recommendations = int(data.get('n_recommendations', 10))

        if not system:
            print("❌ Система не загружена!")
            return jsonify({'error': 'System not loaded'}), 500

        print(f"🎬 Рекомендации для: {movie_title}")
        recommendations = system.get_recommendations(movie_title, n_recommendations)

        if recommendations is None:
            suggestions = system.get_search_suggestions(movie_title)
            print(f"   ⚠️ Фильм не найден. Подсказки: {suggestions}")
            return jsonify({
                'error': 'Movie not found',
                'suggestions': suggestions
            }), 404

        recommendations_list = []
        for idx, row in recommendations.iterrows():
            recommendations_list.append({
                'title': str(row['title']),
                'release_date': str(row['release_date'])[:4] if pd.notna(row['release_date']) else 'N/A',
                'vote_average': float(row['vote_average']) if pd.notna(row['vote_average']) else 0,
                'popularity': float(row['popularity']) if pd.notna(row['popularity']) else 0,
                'genres': str(row['genres_text']),
                'overview': str(row['overview'])[:200] + '...' if len(str(row['overview'])) > 200 else str(
                    row['overview']),
                'similarity_score': float(row['similarity_score'])
            })

        print(f"   ✅ Получено рекомендаций: {len(recommendations_list)}")
        return jsonify({'recommendations': recommendations_list})

    except Exception as e:
        print(f"❌ ОШИБКА в /api/recommendations: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict-revenue', methods=['POST'])
def predict_revenue_api():
    """API для предсказания дохода фильма с выбором модели (Gradient Boosting или Decision Tree)"""

    try:
        data = request.json
        movie_title = data.get('movie_title', '').strip()
        model_type = data.get('model_type', 'gradient_boosting')  # 'gradient_boosting' или 'tree'

        if not movie_title:
            return jsonify({
                'error': 'Movie title is required'
            }), 400

        print(f"💰 Прогноз дохода для: {movie_title} (модель: {model_type})")

        # Выбираем модель
        if model_type == 'tree':
            if not revenue_predictor_tree:
                return jsonify({
                    'error': 'Revenue predictor (Tree) not loaded'
                }), 500
            predictor = revenue_predictor_tree
        else:  # gradient_boosting
            if not revenue_predictor:
                return jsonify({
                    'error': 'Revenue predictor (Gradient Boosting) not loaded'
                }), 500
            predictor = revenue_predictor

        result = predictor.predict_by_title(movie_title)

        if result is None:
            suggestions = system.get_search_suggestions(movie_title) if system else []

            return jsonify({
                'error': 'Movie not found in dataset',
                'suggestions': suggestions
            }), 404

        # Расчет ROI
        roi = (
            ((result['predicted_revenue'] / result['budget']) - 1) * 100
            if result['budget'] > 0 else 0
        )

        response_data = {
            'movie': {
                'title': result['title'],
                'release_date': result['release_date'],
                'genres': result['genres'],
                'vote_average': result['vote_average'],
                'popularity': result['popularity']
            },

            'budget': result['budget'],
            'predicted_revenue': result['predicted_revenue'],
            'actual_revenue': result['actual_revenue'],
            'roi': roi,

            'cast_count': result['cast_count'],
            'runtime': result['runtime'],
            'vote_count': result['vote_count'],

            # МЕТРИКИ МОДЕЛИ
            'model_metrics': predictor.metrics,
            'model_type': model_type
        }

        print(f"   ✅ Прогноз: ${result['predicted_revenue']:,.0f}")

        return jsonify(response_data)

    except Exception as e:
        print(f"❌ ОШИБКА в /api/predict-revenue: {e}")
        traceback.print_exc()

        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/classify-movie', methods=['POST'])
def classify_movie_api():
    """API для классификации фильма (успешный или нет) с выбором модели"""

    try:
        data = request.json
        movie_title = data.get('movie_title', '').strip()
        model_type = data.get('model_type', 'lr_knn')  # 'lr_knn', 'tree', или 'ensemble'
        use_model = data.get('model', 'knn')  # для lr_knn: 'knn' или 'logistic'
        ensemble_model = data.get('ensemble_model',
                                  'random_forest')  # для ensemble: 'random_forest' или 'gradient_boosting'

        if not movie_title:
            return jsonify({
                'error': 'Movie title is required'
            }), 400

        print(f"🎬 Классификация фильма: {movie_title} (тип: {model_type})")

        # Выбираем модель
        if model_type == 'ensemble':
            if not ensemble_classifier:
                return jsonify({
                    'error': 'Ensemble classifier not loaded'
                }), 500

            # ИСПОЛЬЗУЕМ МЕТОД АНСАМБЛЕВОГО КЛАССИФИКАТОРА
            result = ensemble_classifier.classify_movie_ensemble(movie_title, ensemble_model)

            if result is None:
                suggestions = system.get_search_suggestions(movie_title) if system else []
                return jsonify({
                    'error': 'Movie not found in dataset',
                    'suggestions': suggestions
                }), 404

            metrics = ensemble_classifier.metrics_rf if ensemble_model == 'random_forest' else ensemble_classifier.metrics_gb

        elif model_type == 'tree':
            if not movie_classifier_tree:
                return jsonify({
                    'error': 'Movie classifier (Tree) not loaded'
                }), 500

            result = movie_classifier_tree.classify_movie(movie_title)
            metrics = movie_classifier_tree.metrics
            use_model = 'Decision Tree'
        else:  # lr_knn
            if not movie_classifier:
                return jsonify({
                    'error': 'Movie classifier (LR+kNN) not loaded'
                }), 500

            if use_model not in ['knn', 'logistic']:
                use_model = 'knn'

            result = movie_classifier.classify_movie(movie_title, use_model)
            metrics = movie_classifier.metrics_logistic if use_model == 'logistic' else movie_classifier.metrics_knn

        if result is None:
            suggestions = system.get_search_suggestions(movie_title) if system else []

            return jsonify({
                'error': 'Movie not found in dataset',
                'suggestions': suggestions
            }), 404

        response_data = {
            'movie': {
                'title': result['title'],
                'release_date': result['release_date'],
                'genres': result['genres'],
                'vote_average': result['vote_average'],
                'vote_count': result['vote_count'],
                'popularity': result['popularity'],
                'runtime': result['runtime'],
                'budget': result['budget'],
                'cast_count': result['cast_count']
            },
            'classification': {
                'is_successful': result['is_successful'],
                'is_successful_text': result['is_successful_text'],
                'probability_successful': result.get('probability_successful'),
                'probability_unsuccessful': result.get('probability_unsuccessful'),
                'model_used': result.get('model_used', use_model)
            },
            'model_metrics': metrics,
            'model_type': model_type
        }

        print(f"   ✅ Результат: {result['is_successful_text']}")

        return jsonify(response_data)

    except Exception as e:
        print(f"❌ ОШИБКА в /api/classify-movie: {e}")
        traceback.print_exc()

        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/models-info', methods=['GET'])
def models_info():
    """API для получения информации о загруженных моделях"""
    try:
        models_loaded = {
            'recommendation_system': system is not None,
            'revenue_predictor_gb': revenue_predictor is not None,
            'revenue_predictor_tree': revenue_predictor_tree is not None,
            'movie_classifier_lr_knn': movie_classifier is not None,
            'movie_classifier_tree': movie_classifier_tree is not None,
            'ensemble_classifier': ensemble_classifier is not None
        }

        info = {
            'models': models_loaded,
            'available_classification_models': [
                'lr_knn' if movie_classifier else None,
                'tree' if movie_classifier_tree else None,
                'ensemble' if ensemble_classifier else None
            ],
            'available_revenue_models': [
                'gradient_boosting' if revenue_predictor else None,
                'tree' if revenue_predictor_tree else None
            ]
        }

        # Фильтруем None значения
        info['available_classification_models'] = [m for m in info['available_classification_models'] if m]
        info['available_revenue_models'] = [m for m in info['available_revenue_models'] if m]

        return jsonify(info)

    except Exception as e:
        print(f"❌ ОШИБКА в /api/models-info: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Обработка ошибки 404"""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Обработка ошибки 500"""
    print(f"❌ ОШИБКА 500: {error}")
    traceback.print_exc()
    return jsonify({'error': 'Internal server error'}), 500


# ============ ЗАПУСК ============

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🌐 ЗАПУСК СЕРВЕРА FLASK")
    print("=" * 70 + "\n")

    print("📌 Доступные маршруты:")
    print("   GET  / - Главная страница")
    print("   GET  /api/search - Поиск подсказок по названию фильма")
    print("   POST /api/recommendations - Получение рекомендаций")
    print("   POST /api/predict-revenue - Прогноз доходов (Gradient Boosting или Decision Tree)")
    print("   POST /api/classify-movie - Классификация успешности (LR+kNN, Decision Tree или Ensemble)")
    print("   GET  /api/models-info - Информация о загруженных моделях")
    print()

    app.run(debug=True, host='0.0.0.0', port=5000)
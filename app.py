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

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)


class MovieRecommendationSystem:
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

print("=" * 60)
print("🎬 MOVIEMATCH - Загрузка системы")
print("=" * 60 + "\n")

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

system = None
try:
    system = MovieRecommendationSystem(movies_path, keywords_path)
    print("✅ Система рекомендаций загружена успешно!\n")
except Exception as e:
    print(f"❌ ОШИБКА при загрузке системы: {e}")
    traceback.print_exc()
    print()

# Инициализация предсказателя доходов
revenue_predictor = None
model_file = 'revenue_model.pkl'

print("=" * 60)
print("💰 Инициализация предсказателя доходов")
print("=" * 60 + "\n")

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

    print("✅ Предсказатель доходов инициализирован!\n")
except Exception as e:
    print(f"❌ ОШИБКА при инициализации предсказателя: {e}")
    traceback.print_exc()
    revenue_predictor = None
    print()

# Инициализация классификатора фильмов
movie_classifier = None
logistic_model_file = 'logistic_model.pkl'
knn_model_file = 'knn_model.pkl'

print("=" * 60)
print("🎬 Инициализация классификатора фильмов")
print("=" * 60 + "\n")

try:
    # Создаем экземпляр классификатора
    movie_classifier = MovieClassifier(movies_path, credits_path)

    # Проверяем наличие сохраненных моделей
    if os.path.exists(logistic_model_file) and os.path.exists(knn_model_file):
        print(f"📂 Найдены сохраненные модели классификации")
        print(f"   📁 {logistic_model_file}")
        print(f"   📁 {knn_model_file}")
        try:
            movie_classifier.load_models(logistic_model_file, knn_model_file)
            print("✅ Модели успешно загружены!\n")
        except Exception as load_error:
            print(f"⚠️ Ошибка при загрузке моделей: {load_error}")
            print(f"🔨 Переобучаем модели...\n")
            metrics = movie_classifier.train_models()
            movie_classifier.save_models(logistic_model_file, knn_model_file)
    else:
        print(f"🔨 Обученные модели не найдены. Обучаем новые...\n")
        metrics = movie_classifier.train_models()
        movie_classifier.save_models(logistic_model_file, knn_model_file)

    print("✅ Классификатор фильмов инициализирован!\n")
except Exception as e:
    print(f"❌ ОШИБКА при инициализации классификатора: {e}")
    traceback.print_exc()
    movie_classifier = None
    print()

print("=" * 60)
print("🚀 ПРИЛОЖЕНИЕ ГОТОВО К РАБОТЕ")
print("=" * 60)
print("\n📍 Откройте браузер: http://127.0.0.1:5000\n")


# ============ МАРШРУТЫ ============

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')


@app.route('/api/search', methods=['GET'])
def search_suggestions():
    """API для поиска подсказок"""
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

        print(f"🔍 Поиск: '{query}'")
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
    """API для получения рекомендаций"""
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
    """API для предсказания дохода по названию фильма"""

    try:
        if not revenue_predictor:
            return jsonify({
                'error': 'Revenue predictor not loaded'
            }), 500

        data = request.json
        movie_title = data.get('movie_title', '').strip()

        if not movie_title:
            return jsonify({
                'error': 'Movie title is required'
            }), 400

        print(f"💰 Прогноз дохода для: {movie_title}")

        result = revenue_predictor.predict_by_title(movie_title)

        if result is None:
            suggestions = system.get_search_suggestions(movie_title) if system else []

            return jsonify({
                'error': 'Movie not found in dataset',
                'suggestions': suggestions
            }), 404

        # ROI
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
            'model_metrics': revenue_predictor.metrics
        }

        print(f"   ✅ Прогноз: ${result['predicted_revenue']:,.0f}")
        print(f"   📈 MSE: {revenue_predictor.metrics['mse']:,.0f}")

        return jsonify(response_data)

    except Exception as e:
        print(f"❌ ОШИБКА в /api/predict-revenue: {e}")
        traceback.print_exc()

        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/classify-movie', methods=['POST'])
def classify_movie_api():
    """API для классификации фильма (успешный или нет)"""

    try:
        if not movie_classifier:
            return jsonify({
                'error': 'Movie classifier not loaded'
            }), 500

        data = request.json
        movie_title = data.get('movie_title', '').strip()
        use_model = data.get('model', 'knn').lower()  # 'knn' или 'logistic'

        if not movie_title:
            return jsonify({
                'error': 'Movie title is required'
            }), 400

        # Валидация модели
        if use_model not in ['knn', 'logistic']:
            use_model = 'knn'
            print(f"⚠️ Неизвестная модель, используется knn по умолчанию")

        print(f"🎬 Классификация фильма: '{movie_title}' (модель: {use_model})")

        # Вызываем метод классификации
        result = movie_classifier.classify_movie(movie_title, use_model)

        if result is None:
            suggestions = system.get_search_suggestions(movie_title) if system else []
            print(f"   ⚠️ Фильм не найден в датасете")

            return jsonify({
                'error': 'Movie not found in dataset',
                'suggestions': suggestions
            }), 404

        # Выбираем правильные метрики в зависимости от модели
        model_metrics = (
            movie_classifier.metrics_logistic if use_model == 'logistic'
            else movie_classifier.metrics_knn
        )

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
                'probability_successful': result['probability_successful'],
                'probability_unsuccessful': result['probability_unsuccessful'],
                'model_used': result['model_used']
            },
            'model_metrics': model_metrics
        }

        print(f"   ✅ Результат: {result['is_successful_text']}")
        if result['probability_successful'] is not None:
            print(f"   📊 Вероятность успешности: {result['probability_successful']:.2%}")

        return jsonify(response_data)

    except ValueError as ve:
        print(f"❌ ОШИБКА ЗНАЧЕНИЯ в /api/classify-movie: {ve}")
        return jsonify({
            'error': str(ve)
        }), 400

    except Exception as e:
        print(f"❌ ОШИБКА в /api/classify-movie: {e}")
        traceback.print_exc()

        return jsonify({
            'error': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    print(f"❌ ОШИБКА 500: {error}")
    traceback.print_exc()
    return jsonify({'error': 'Internal server error'}), 500


# ============ ЗАПУСК ============

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🌐 Запуск сервера Flask...")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)

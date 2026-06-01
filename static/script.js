// ============ ЭЛЕМЕНТЫ DOM ============

// Элементы вкладок
const tabButtons = document.querySelectorAll('.tab-button');
const tabContents = document.querySelectorAll('.tab-content');

// Элементы рекомендаций
const movieInput = document.getElementById('movieInput');
const countInput = document.getElementById('countInput');
const searchForm = document.getElementById('searchForm');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const results = document.getElementById('results');
const emptyState = document.getElementById('empty-state');
const recommendationsList = document.getElementById('recommendationsList');
const suggestionsDropdown = document.getElementById('suggestions');

// Элементы прогноза
const predictionForm = document.getElementById('predictionForm');
const predictionMovieInput = document.getElementById('predictionMovieInput');
const predictionSuggestions = document.getElementById('predictionSuggestions');
const predictionLoading = document.getElementById('predictionLoading');
const predictionError = document.getElementById('predictionError');
const predictionResult = document.getElementById('predictionResult');
const emptyPrediction = document.getElementById('empty-prediction');
const revenueModelSelect = document.getElementById('revenueModelSelect');

// Элементы классификации
const classificationForm = document.getElementById('classificationForm');
const classificationMovieInput = document.getElementById('classificationMovieInput');
const classificationSuggestions = document.getElementById('classificationSuggestions');
const classificationLoading = document.getElementById('classificationLoading');
const classificationError = document.getElementById('classificationError');
const classificationResult = document.getElementById('classificationResult');
const emptyClassification = document.getElementById('empty-classification');
const classificationModelSelect = document.getElementById('classificationModelSelect');
const subModelSelector = document.getElementById('subModelSelector');
const modelSelect = document.getElementById('modelSelect');

// Переменные
let debounceTimer;
let revenueChart = null;

// ============ ИНИЦИАЛИЗАЦИЯ ============

document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
});

function initializeEventListeners() {
    // Вкладки
    tabButtons.forEach(button => {
        button.addEventListener('click', handleTabClick);
    });

    // Рекомендации
    movieInput.addEventListener('input', handleMovieInput);
    searchForm.addEventListener('submit', handleSearchSubmit);
    document.addEventListener('click', handleDocumentClick);

    // Прогноз
    predictionMovieInput.addEventListener('input', handlePredictionInput);
    predictionForm.addEventListener('submit', handlePredictionSubmit);
    revenueModelSelect.addEventListener('change', () => console.log('Model selected:', revenueModelSelect.value));

    // Классификация
    classificationMovieInput.addEventListener('input', handleClassificationInput);
    classificationForm.addEventListener('submit', handleClassificationSubmit);
    classificationModelSelect.addEventListener('change', handleClassificationModelChange);
}

// ============ УПРАВЛЕНИЕ ВКЛАДКАМИ ============

function handleTabClick(e) {
    const button = e.currentTarget;
    const tabName = button.getAttribute('data-tab');

    // Удаляем активность со всех вкладок
    tabButtons.forEach(b => b.classList.remove('active'));
    tabContents.forEach(c => c.classList.remove('active'));

    // Добавляем активность выбранной вкладке
    button.classList.add('active');
    document.getElementById(tabName).classList.add('active');

    console.log(`📑 Переключение на вкладку: ${tabName}`);
}

// ============ РЕКОМЕНДАЦИИ ФИЛЬМОВ ============

/**
 * Обработка ввода названия фильма для рекомендаций
 */
function handleMovieInput(e) {
    clearTimeout(debounceTimer);
    const query = e.target.value.trim();

    if (query.length < 2) {
        suggestionsDropdown.innerHTML = '';
        suggestionsDropdown.classList.remove('active');
        return;
    }

    debounceTimer = setTimeout(() => {
        fetchSuggestions(query);
    }, 300);
}

/**
 * Получение подсказок для рекомендаций
 */
async function fetchSuggestions(query) {
    try {
        console.log(`🔍 Поиск подсказок: ${query}`);

        const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);

        if (!response.ok) {
            console.error(`❌ Ошибка HTTP: ${response.status}`);
            return;
        }

        const data = await response.json();

        if (data.error) {
            console.error(`❌ Ошибка сервера: ${data.error}`);
            suggestionsDropdown.classList.remove('active');
            return;
        }

        if (data.suggestions && data.suggestions.length > 0) {
            console.log(`✅ Найдено подсказок: ${data.suggestions.length}`);
            displaySuggestions(data.suggestions);
        } else {
            console.log('⚠️ Подсказки не найдены');
            suggestionsDropdown.classList.remove('active');
        }
    } catch (err) {
        console.error('❌ Ошибка при получении подсказок:', err);
    }
}
/**
 * Отображение подсказок
 */
function displaySuggestions(suggestions) {
    suggestionsDropdown.innerHTML = suggestions
        .map(s => `<div class="suggestion-item">${escapeHtml(s)}</div>`)
        .join('');

    suggestionsDropdown.classList.add('active');

    // Добавляем слушатели на подсказки
    document.querySelectorAll('.suggestion-item').forEach(item => {
        item.addEventListener('click', () => {
            movieInput.value = item.textContent;
            suggestionsDropdown.classList.remove('active');
        });
    });
}

/**
 * Закрытие подсказок при клике снаружи
 */
function handleDocumentClick(e) {
    if (!e.target.closest('.search-input-wrapper')) {
        suggestionsDropdown.classList.remove('active');
    }
    if (!e.target.closest('.search-input-wrapper-prediction')) {
        predictionSuggestions.classList.remove('active');
        classificationSuggestions.classList.remove('active');
    }
}

/**
 * Обработка отправки формы поиска рекомендаций
 */
async function handleSearchSubmit(e) {
    e.preventDefault();

    const movieTitle = movieInput.value.trim();
    const n_recommendations = parseInt(countInput.value);

    if (!movieTitle) {
        showError('❌ Пожалуйста, введите название фильма');
        return;
    }

    console.log(`🎬 Поиск рекомендаций для: ${movieTitle} (${n_recommendations} шт)`);
    await fetchRecommendations(movieTitle, n_recommendations);
}

/**
 * Получение рекомендаций фильмов
 */
async function fetchRecommendations(movieTitle, n_recommendations) {
    showLoading(true);
    hideError();
    hideResults();
    hideEmpty();

    try {
        const response = await fetch('/api/recommendations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                movie_title: movieTitle,
                n_recommendations: n_recommendations
            })
        });

        const data = await response.json();

        if (!response.ok) {
            console.log(`⚠️ Фильм не найден: ${movieTitle}`);
            showError(`❌ Фильм "${movieTitle}" не найден`, data.suggestions);
            showLoading(false);
            return;
        }

        console.log(`✅ Получено рекомендаций: ${data.recommendations.length}`);
        displayRecommendations(data.recommendations);
        showLoading(false);
    } catch (err) {
        console.error('❌ Ошибка при получении рекомендаций:', err);
        showError('⚠️ Ошибка при получении рекомендаций. Попробуйте позже.');
        showLoading(false);
    }
}

/**
 * Отображение рекомендаций
 */
function displayRecommendations(recommendations) {
    if (!recommendations || recommendations.length === 0) {
        console.log('⚠️ Нет рекомендаций');
        showEmpty();
        return;
    }

    recommendationsList.innerHTML = recommendations
        .map((movie, index) => createMovieCard(movie, index))
        .join('');

    showResults();
}

/**
 * Создание карточки фильма
 */
function createMovieCard(movie, index) {
    const genres = movie.genres
        .split(' ')
        .filter(g => g.length > 0)
        .slice(0, 3);

    const genreTags = genres
        .map(g => `<div class="genre-tag">${escapeHtml(g)}</div>`)
        .join('');

    const overview = movie.overview || 'Описание недоступно';
    const truncatedOverview = overview.length > 200 ? overview.substring(0, 200) + '...' : overview;

    return `
        <div class="movie-card" style="animation: slideIn 0.3s ease forwards; animation-delay: ${index * 0.05}s; opacity: 0;">
            <div class="movie-card-header">
                <div class="movie-title">${escapeHtml(movie.title)}</div>
                <div class="movie-year">${movie.release_date}</div>
            </div>

            <div class="movie-meta">
                <div class="meta-item">
                    <i class="fas fa-star"></i>
                    <span class="movie-rating">${movie.vote_average.toFixed(1)}/10</span>
                </div>
                <div class="meta-item">
                    <i class="fas fa-fire"></i>
                    <span class="movie-popularity">${movie.popularity.toFixed(0)}</span>
                </div>
            </div>

            <div class="movie-genres">
                ${genreTags}
            </div>

            <div class="movie-overview">
                ${escapeHtml(truncatedOverview)}
            </div>

            <div class="movie-similarity">
                <span class="similarity-label">Совпадение:</span>
                <span class="similarity-value">${(movie.similarity_score * 100).toFixed(1)}%</span>
            </div>
        </div>
    `;
}

// ============ ПРОГНОЗ ДОХОДОВ ============

/**
 * Обработка ввода названия фильма для прогноза
 */
function handlePredictionInput(e) {
    clearTimeout(debounceTimer);
    const query = e.target.value.trim();

    if (query.length < 2) {
        predictionSuggestions.innerHTML = '';
        predictionSuggestions.classList.remove('active');
        return;
    }

    debounceTimer = setTimeout(() => {
        fetchPredictionSuggestions(query);
    }, 300);
}

/**
 * Получение подсказок для прогноза
 */
async function fetchPredictionSuggestions(query) {
    try {
        console.log(`🔍 Поиск подсказок прогноза: ${query}`);

        const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
        const data = await response.json();

        if (data.suggestions && data.suggestions.length > 0) {
            console.log(`✅ Найдено подсказок прогноза: ${data.suggestions.length}`);
            displayPredictionSuggestions(data.suggestions);
        } else {
            console.log('⚠️ Подсказки прогноза не найдены');
            predictionSuggestions.classList.remove('active');
        }
    } catch (err) {
        console.error('❌ Ошибка при получении подсказок прогноза:', err);
    }
}

/**
 * Отображение подсказок для прогноза
 */
function displayPredictionSuggestions(suggestions) {
    predictionSuggestions.innerHTML = suggestions
        .map(s => `<div class="suggestion-item">${escapeHtml(s)}</div>`)
        .join('');

    predictionSuggestions.classList.add('active');

    document.querySelectorAll('#predictionSuggestions .suggestion-item').forEach(item => {
        item.addEventListener('click', () => {
            predictionMovieInput.value = item.textContent;
            predictionSuggestions.classList.remove('active');
        });
    });
}

/**
 * Обработка отправки формы прогноза
 */
async function handlePredictionSubmit(e) {
    e.preventDefault();

    const movieTitle = predictionMovieInput.value.trim();
    const modelType = revenueModelSelect.value;

    if (!movieTitle) {
        showPredictionError('❌ Пожалуйста, введите название фильма');
        return;
    }

    console.log(`💰 Прогноз доходов для: ${movieTitle} (модель: ${modelType})`);
    await fetchRevenuePredict(movieTitle, modelType);
}

/**
 * Получение прогноза доходов
 */
async function fetchRevenuePredict(movieTitle, modelType) {
    showPredictionLoading(true);
    hidePredictionError();
    hidePredictionResult();

    try {
        const response = await fetch('/api/predict-revenue', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                movie_title: movieTitle,
                model_type: modelType
            })
        });

        const data = await response.json();

        if (!response.ok) {
            console.log(`⚠️ Фильм для прогноза не найден: ${movieTitle}`);
            showPredictionError(data.error || 'Фильм не найден', data.suggestions);
            showPredictionLoading(false);
            return;
        }

        console.log(`✅ Получен прогноз для: ${data.movie.title}`);
        displayPredictionResult(data);
        showPredictionLoading(false);
    } catch (err) {
        console.error('❌ Ошибка при получении прогноза:', err);
        showPredictionError('❌ Ошибка при получении прогноза');
        showPredictionLoading(false);
    }
}

/**
 * Отображение результатов прогноза
 */
function displayPredictionResult(data) {
    const movie = data.movie;
    const budget = data.budget;
    const predictedRevenue = data.predicted_revenue;
    const actualRevenue = data.actual_revenue;
    const roi = data.roi;

    console.log(`📊 Результаты:
        Бюджет: $${formatNumber(budget)}
        Прогноз: $${formatNumber(predictedRevenue)}
        Факт: $${formatNumber(actualRevenue)}
        ROI: ${roi.toFixed(1)}%`);

    // Заполняем данные
    document.getElementById('resultTitle').textContent = `${movie.title}`;
    document.getElementById('resultSubtitle').textContent = `${movie.genres} • ${movie.release_date}`;

    document.getElementById('budgetValue').textContent = `$${formatNumber(budget)}`;
    document.getElementById('predictedRevenue').textContent = `$${formatNumber(predictedRevenue)}`;
    document.getElementById('roiValue').textContent = `${roi.toFixed(1)}%`;
    document.getElementById('voteValue').textContent = `${movie.vote_average.toFixed(1)}/10`;
    document.getElementById('modelTypeValue').textContent = data.model_type === 'tree' ? 'Decision Tree' : 'Gradient Boosting';

    // Метрики модели
    if (data.model_metrics) {
        document.getElementById('mseValue').textContent =
            '$' + formatNumber(data.model_metrics.rmse || data.model_metrics.test_rmse);

        console.log('📈 Метрики модели:', data.model_metrics);
    }

    document.getElementById('genreValue').textContent = movie.genres || '-';
    document.getElementById('yearValue').textContent = movie.release_date || '-';
    document.getElementById('runtimeValue').textContent = `${data.runtime} мин` || '-';
    document.getElementById('castValue').textContent = `${data.cast_count} актеров` || '-';
    document.getElementById('voteCountValue').textContent = formatNumber(data.vote_count);
    document.getElementById('popularityValue').textContent = movie.popularity.toFixed(1);
    document.getElementById('actualRevenueValue').textContent = actualRevenue > 0 ? `$${formatNumber(actualRevenue)}` : 'Нет данных';

    // Рисуем график
    drawRevenueChart(budget, predictedRevenue, actualRevenue);

    // Показываем результаты
    showPredictionResult();
}

/**
 * Рисование графика доходов
 */
function drawRevenueChart(budget, predicted, actual) {
    const ctx = document.getElementById('revenueChart').getContext('2d');

    // Удаляем старый график
    if (revenueChart) {
        revenueChart.destroy();
    }

    const datasets = [
        {
            label: 'Бюджет',
            data: [budget, 0, 0],
            backgroundColor: 'rgba(78, 205, 196, 0.7)',
            borderColor: 'rgba(78, 205, 196, 1)',
            borderWidth: 2,
            borderRadius: 8
        },
        {
            label: 'Прогноз доходов',
            data: [0, predicted, 0],
            backgroundColor: 'rgba(255, 107, 107, 0.7)',
            borderColor: 'rgba(255, 107, 107, 1)',
            borderWidth: 2,
            borderRadius: 8
        }
    ];

    // Добавляем фактический доход, если есть
    if (actual > 0) {
        datasets.push({
            label: 'Фактический доход',
            data: [0, 0, actual],
            backgroundColor: 'rgba(255, 230, 109, 0.7)',
            borderColor: 'rgba(255, 230, 109, 1)',
            borderWidth: 2,
            borderRadius: 8
        });
    }

    const labels = ['Бюджет', 'Прогноз', actual > 0 ? 'Факт' : ''];

    revenueChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels.filter(l => l),
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            indexAxis: 'y',
            scales: {
                x: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return '$' + (value / 1000000).toFixed(0) + 'B';
                        },
                        color: '#B8BCC8'
                    },
                    grid: {
                        color: 'rgba(58, 68, 82, 0.3)'
                    }
                },
                y: {
                    ticks: {
                        color: '#B8BCC8'
                    },
                    grid: {
                        display: false
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#B8BCC8',
                        padding: 15,
                        font: {
                            size: 12
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.7)',
                    titleColor: '#FFF',
                    bodyColor: '#FFF',
                    borderColor: '#FF6B6B',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            return '$' + formatNumber(context.parsed.x);
                        }
                    }
                }
            }
        }
    });
}

// ============ КЛАССИФИКАЦИЯ ФИЛЬМОВ ============

/**
 * Переключение видимости подмодели для LR+kNN и показ селектора для Ensemble
 */
function handleClassificationModelChange(e) {
    const modelType = e.target.value;
    if (modelType === 'lr_knn') {
        subModelSelector.style.display = 'block';
    } else {
        subModelSelector.style.display = 'none';
    }
}

/**
 * Обработка ввода названия фильма для классификации
 */
function handleClassificationInput(e) {
    clearTimeout(debounceTimer);
    const query = e.target.value.trim();

    if (query.length < 2) {
        classificationSuggestions.innerHTML = '';
        classificationSuggestions.classList.remove('active');
        return;
    }

    debounceTimer = setTimeout(() => {
        fetchClassificationSuggestions(query);
    }, 300);
}

/**
 * Получение подсказок для классификации
 */
async function fetchClassificationSuggestions(query) {
    try {
        console.log(`🔍 Поиск подсказок классификации: ${query}`);

        const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
        const data = await response.json();

        if (data.suggestions && data.suggestions.length > 0) {
            console.log(`✅ Найдено подсказок классификации: ${data.suggestions.length}`);
            displayClassificationSuggestions(data.suggestions);
        } else {
            console.log('⚠️ Подсказки классификации не найдены');
            classificationSuggestions.classList.remove('active');
        }
    } catch (err) {
        console.error('❌ Ошибка при получении подсказок классификации:', err);
    }
}

/**
 * Отображение подсказок для классификации
 */
function displayClassificationSuggestions(suggestions) {
    classificationSuggestions.innerHTML = suggestions
        .map(s => `<div class="suggestion-item">${escapeHtml(s)}</div>`)
        .join('');

    classificationSuggestions.classList.add('active');

    document.querySelectorAll('#classificationSuggestions .suggestion-item').forEach(item => {
        item.addEventListener('click', () => {
            classificationMovieInput.value = item.textContent;
            classificationSuggestions.classList.remove('active');
        });
    });
}

/**
 * Обработка отправки формы классификации
 */
async function handleClassificationSubmit(e) {
    e.preventDefault();

    const movieTitle = classificationMovieInput.value.trim();
    const modelType = classificationModelSelect.value;
    const useModel = modelSelect.value;

    if (!movieTitle) {
        showClassificationError('❌ Пожалуйста, введите название фильма');
        return;
    }

    console.log(`🎬 Классификация фильма: ${movieTitle} (тип: ${modelType})`);
    await fetchClassification(movieTitle, modelType, useModel);
}

/**
 * Получение классификации
 */
async function fetchClassification(movieTitle, modelType, useModel) {
    showClassificationLoading(true);
    hideClassificationError();
    hideClassificationResult();

    try {
        const requestData = {
            movie_title: movieTitle,
            model_type: modelType,
            model: useModel
        };

        // Если это ensemble модель, добавляем параметр ансамбля
        if (modelType === 'ensemble') {
            requestData.ensemble_model = useModel === 'logistic' ? 'gradient_boosting' : 'random_forest';
        }

        const response = await fetch('/api/classify-movie', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });

        const data = await response.json();

        if (!response.ok) {
            console.log(`⚠️ Фильм для классификации не найден: ${movieTitle}`);
            showClassificationError(data.error || 'Фильм не найден', data.suggestions);
            showClassificationLoading(false);
            return;
        }

        console.log(`✅ Получена классификация для: ${data.movie.title}`);
        displayClassificationResult(data);
        showClassificationLoading(false);
    } catch (err) {
        console.error('❌ Ошибка при получении классификации:', err);
        showClassificationError('❌ Ошибка при получении классификации');
        showClassificationLoading(false);
    }
}

/**
 * Отображение результатов классификации
 */
function displayClassificationResult(data) {
    const movie = data.movie;
    const classification = data.classification;

    console.log(`🎬 Результаты классификации:
        Название: ${movie.title}
        Успешный: ${classification.is_successful}
        Модель: ${classification.model_used}`);

    // Заполняем данные
    document.getElementById('classResultTitle').textContent = `${movie.title}`;
    document.getElementById('classResultSubtitle').textContent = `${movie.genres} • ${movie.release_date}`;

    // Значок успешности
    const badge = document.getElementById('classificationBadge');
    if (classification.is_successful) {
        badge.innerHTML = '✅ УСПЕШНЫЙ ФИЛЬМ';
        badge.style.backgroundColor = '#4ECDC4';
    } else {
        badge.innerHTML = '❌ МЕНЕЕ УСПЕШНЫЙ';
        badge.style.backgroundColor = '#FF6B6B';
    }

    // Метрики фильма
    document.getElementById('classVoteValue').textContent = `${movie.vote_average.toFixed(1)}/10`;
    document.getElementById('classPopularityValue').textContent = movie.popularity.toFixed(1);
    document.getElementById('classVoteCountValue').textContent = formatNumber(movie.vote_count);
    document.getElementById('classCastValue').textContent = movie.cast_count;

    // Детали
    document.getElementById('classGenreValue').textContent = movie.genres || '-';
    document.getElementById('classYearValue').textContent = movie.release_date || '-';
    document.getElementById('classRuntimeValue').textContent = `${movie.runtime} мин` || '-';
    document.getElementById('classBudgetValue').textContent = `$${formatNumber(movie.budget)}` || '-';
    document.getElementById('classModelValue').textContent = classification.model_used;

    // Вероятность
    if (classification.probability_successful !== null && classification.probability_unsuccessful !== null) {
        const probSuccess = (classification.probability_successful * 100).toFixed(1);
        document.getElementById('classProbabilityValue').textContent = `${probSuccess}%`;
    } else {
        document.getElementById('classProbabilityValue').textContent = 'N/A';
    }

    // Метрики модели
    if (data.model_metrics) {
        const metrics = data.model_metrics;

        // Для обеих типов моделей
        document.getElementById('metricAccuracy').textContent =
            (metrics.test_accuracy || metrics.accuracy || '-').toFixed(3);
        document.getElementById('metricPrecision').textContent =
            (metrics.test_precision || metrics.precision || '-').toFixed(3);
        document.getElementById('metricRecall').textContent =
            (metrics.test_recall || metrics.recall || '-').toFixed(3);
        document.getElementById('metricF1').textContent =
            (metrics.test_f1 || metrics.f1 || '-').toFixed(3);
        document.getElementById('metricROC').textContent =
            (metrics.test_roc_auc || metrics.roc_auc || '-').toFixed(3);

        console.log('📈 Метрики модели:', data.model_metrics);
    }

    showClassificationResult();
}

// ============ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ============

/**
 * Форматирование числа с разделителем тысяч
 */
function formatNumber(num) {
    return Math.round(num).toLocaleString('en-US');
}

/**
 * Экранирование HTML специальных символов
 */
function escapeHtml(text) {
    if (!text) return '';

    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// ============ ФУНКЦИИ ВИДИМОСТИ (Рекомендации) ============

function showLoading(show) {
    loading.classList.toggle('hidden', !show);
}

function showResults() {
    results.classList.remove('hidden');
    emptyState.classList.add('hidden');
}

function showEmpty() {
    results.classList.add('hidden');
    emptyState.classList.remove('hidden');
}

function hideResults() {
    results.classList.add('hidden');
}

function hideEmpty() {
    emptyState.classList.add('hidden');
}

function showError(message, suggestions = null) {
    let fullMessage = message;
    if (suggestions && suggestions.length > 0) {
        fullMessage += `\n💡 Может быть вы имеете в виду: ${suggestions.slice(0, 3).join(', ')}`;
    }
    error.textContent = fullMessage;
    error.classList.remove('hidden');
}

function hideError() {
    error.classList.add('hidden');
}

// ============ ФУНКЦИИ ВИДИМОСТИ (Прогноз) ============

function showPredictionLoading(show) {
    predictionLoading.classList.toggle('hidden', !show);
}

function showPredictionResult() {
    predictionResult.classList.remove('hidden');
    emptyPrediction.classList.add('hidden');
}

function hidePredictionResult() {
    predictionResult.classList.add('hidden');
    emptyPrediction.classList.remove('hidden');
}

function showPredictionError(message, suggestions = null) {
    let fullMessage = `❌ ${message}`;
    if (suggestions && suggestions.length > 0) {
        fullMessage += `\n💡 Может быть вы имеете в виду: ${suggestions.slice(0, 3).join(', ')}`;
    }
    predictionError.textContent = fullMessage;
    predictionError.classList.remove('hidden');
}

function hidePredictionError() {
    predictionError.classList.add('hidden');
}

// ============ ФУНКЦИИ ВИДИМОСТИ (Классификация) ============

function showClassificationLoading(show) {
    classificationLoading.classList.toggle('hidden', !show);
}

function showClassificationResult() {
    classificationResult.classList.remove('hidden');
    emptyClassification.classList.add('hidden');
}

function hideClassificationResult() {
    classificationResult.classList.add('hidden');
    emptyClassification.classList.remove('hidden');
}

function showClassificationError(message, suggestions = null) {
    let fullMessage = `❌ ${message}`;
    if (suggestions && suggestions.length > 0) {
        fullMessage += `\n💡 Может быть вы имеете в виду: ${suggestions.slice(0, 3).join(', ')}`;
    }
    classificationError.textContent = fullMessage;
    classificationError.classList.remove('hidden');
}

function hideClassificationError() {
    classificationError.classList.add('hidden');
}

// ============ АНИМАЦИИ ============

/**
 * Инициализация анимаций
 */
const animationStyle = document.createElement('style');
animationStyle.textContent = `
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }

    @keyframes spin {
        to {
            transform: rotate(360deg);
        }
    }
`;
document.head.appendChild(animationStyle);

// ============ ЛОГИРОВАНИЕ ============

console.log('%c🎬 MovieMatch приложение загружено!', 'color: #FF6B6B; font-size: 16px; font-weight: bold;');
console.log('%cВкладка "Рекомендации" - получайте похожие фильмы', 'color: #4ECDC4; font-size: 12px;');
console.log('%cВкладка "Прогноз доходов" - предсказывайте кассовые сборы (Gradient Boosting + Decision Tree)', 'color: #FFE66D; font-size: 12px;');
console.log('%cВкладка "Классификация" - определяйте успешность фильмов (LR+kNN + Decision Tree + Ensemble Models)', 'color: #FF6B9D; font-size: 12px;');
    
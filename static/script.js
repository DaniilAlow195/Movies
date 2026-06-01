// ============ ИНИЦИАЛИЗАЦИЯ ============
const API_BASE = 'http://127.0.0.1:5000/api';

// Вкладки
const classificationTabButton = document.getElementById('classification-tab');
const revenueTabButton = document.getElementById('revenue-tab');
const recommendationsTabButton = document.getElementById('recommendations-tab');

const classificationSection = document.getElementById('classification-section');
const revenueSection = document.getElementById('revenue-section');
const recommendationsSection = document.getElementById('recommendations-section');

// ============ УПРАВЛЕНИЕ ВКЛАДКАМИ ============
classificationTabButton.addEventListener('click', () => showTab('classification'));
revenueTabButton.addEventListener('click', () => showTab('revenue'));
recommendationsTabButton.addEventListener('click', () => showTab('recommendations'));

function showTab(tabName) {
    // Скрываем все вкладки
    classificationSection.style.display = 'none';
    revenueSection.style.display = 'none';
    recommendationsSection.style.display = 'none';
    
    // Убираем активный класс
    classificationTabButton.classList.remove('active');
    revenueTabButton.classList.remove('active');
    recommendationsTabButton.classList.remove('active');
    
    // Показываем нужную вкладку
    if (tabName === 'classification') {
        classificationSection.style.display = 'block';
        classificationTabButton.classList.add('active');
    } else if (tabName === 'revenue') {
        revenueSection.style.display = 'block';
        revenueTabButton.classList.add('active');
    } else if (tabName === 'recommendations') {
        recommendationsSection.style.display = 'block';
        recommendationsTabButton.classList.add('active');
    }
}

// ============ ИЗМЕНЕНИЕ МОДЕЛИ КЛАССИФИКАЦИИ ============
document.getElementById('model-type').addEventListener('change', function() {
    const useModelDiv = document.getElementById('use-model-div');
    const ensembleModelDiv = document.getElementById('ensemble-model-div');
    
    // Скрываем все опции
    if (useModelDiv) useModelDiv.style.display = 'none';
    if (ensembleModelDiv) ensembleModelDiv.style.display = 'none';
    
    // Показываем нужные опции
    if (this.value === 'lr_knn') {
        useModelDiv.style.display = 'block';
    } else if (this.value === 'ensemble') {
        ensembleModelDiv.style.display = 'block';
    }
});

// ============ КЛАССИФИКАЦИЯ ФИЛЬМОВ ============
document.getElementById('classify-btn').addEventListener('click', async (e) => {
    e.preventDefault();
    
    const movieTitle = document.getElementById('movie-input').value.trim();
    const modelType = document.getElementById('model-type').value;
    
    if (!movieTitle) {
        alert('Пожалуйста, введите название фильма');
        return;
    }
    
    try {
        document.querySelector('#classification-section .prediction-result').innerHTML = '<p style="text-align: center;">⏳ Анализ...</p>';
        
        let payload = {
            movie_title: movieTitle,
            model_type: modelType
        };
        
        if (modelType === 'lr_knn') {
            payload.model = document.getElementById('use-model').value;
        } else if (modelType === 'ensemble') {
            payload.ensemble_model = document.getElementById('ensemble-model').value;
        }
        
        const response = await fetch(`${API_BASE}/classify-movie`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayClassificationResult(data);
        } else {
            document.querySelector('#classification-section .prediction-result').innerHTML = `<p style="color: red;">❌ Ошибка: ${data.error}</p>`;
        }
    } catch (error) {
        console.error('Ошибка:', error);
        document.querySelector('#classification-section .prediction-result').innerHTML = '<p style="color: red;">❌ Ошибка при классификации</p>';
    }
});

function displayClassificationResult(data) {
    const movie = data.movie;
    const classification = data.classification;
    const metrics = data.model_metrics || {};
    
    let html = `
        <div class="result-card" style="padding: 20px; background: #1e2139; border-radius: 8px; margin-top: 20px;">
            <h3 style="color: #fff; margin-bottom: 10px;">${movie.title}</h3>
            <p style="color: #b8bcc8; margin: 5px 0;"><strong>Год:</strong> ${movie.release_date}</p>
            <p style="color: #b8bcc8; margin: 5px 0;"><strong>Жанр:</strong> ${movie.genres}</p>
            <p style="color: #b8bcc8; margin: 5px 0;"><strong>Рейтинг:</strong> ${movie.vote_average.toFixed(1)}/10</p>
            <p style="color: #b8bcc8; margin: 5px 0;"><strong>Популярность:</strong> ${movie.popularity.toFixed(2)}</p>
            
            <div style="margin-top: 15px; padding: 15px; background: #0f1620; border-radius: 5px;">
                <h4 style="color: #4ECDC4; margin-bottom: 10px;">Результат классификации:</h4>
                <p style="font-size: 18px; font-weight: bold; color: ${classification.is_successful ? '#4ECDC4' : '#FF6B6B'};">
                    ${classification.is_successful_text}
                </p>
                <p style="color: #b8bcc8; margin: 10px 0;"><strong>Модель:</strong> ${classification.model_used}</p>
    `;
    
    if (classification.probability_successful !== null) {
        html += `<p style="color: #b8bcc8;"><strong>Вероятность успеха:</strong> ${(classification.probability_successful * 100).toFixed(1)}%</p>`;
    }
    
    html += `</div>`;
    
    if (metrics && Object.keys(metrics).length > 0) {
        html += `
            <div style="margin-top: 15px; padding: 15px; background: #0f1620; border-radius: 5px;">
                <h4 style="color: #FFE66D; margin-bottom: 10px;">Метрики модели:</h4>
                <p style="color: #b8bcc8;"><strong>Accuracy:</strong> ${((metrics.accuracy || 0) * 100).toFixed(1)}%</p>
        `;
        if (metrics.precision) html += `<p style="color: #b8bcc8;"><strong>Precision:</strong> ${((metrics.precision) * 100).toFixed(1)}%</p>`;
        if (metrics.recall) html += `<p style="color: #b8bcc8;"><strong>Recall:</strong> ${((metrics.recall) * 100).toFixed(1)}%</p>`;
        html += `</div>`;
    }
    
    html += `</div>`;
    document.querySelector('#classification-section .prediction-result').innerHTML = html;
}

// ============ ПРОГНОЗ ДОХОДОВ ============
document.getElementById('revenue-btn').addEventListener('click', async (e) => {
    e.preventDefault();
    
    const movieTitle = document.getElementById('revenue-movie-input').value.trim();
    const modelType = document.getElementById('revenue-model-type').value;
    
    if (!movieTitle) {
        alert('Пожалуйста, введите название фильма');
        return;
    }
    
    try {
        document.querySelector('#revenue-section .prediction-result').innerHTML = '<p style="text-align: center;">⏳ Прогнозирование...</p>';
        
        const response = await fetch(`${API_BASE}/predict-revenue`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                movie_title: movieTitle,
                model_type: modelType
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayRevenueResult(data);
        } else {
            document.querySelector('#revenue-section .prediction-result').innerHTML = `<p style="color: red;">❌ Ошибка: ${data.error}</p>`;
        }
    } catch (error) {
        console.error('Ошибка:', error);
        document.querySelector('#revenue-section .prediction-result').innerHTML = '<p style="color: red;">❌ Ошибка при прогнозировании</p>';
    }
});

function displayRevenueResult(data) {
    const movie = data.movie;
    const budget = data.budget;
    const predictedRevenue = data.predicted_revenue;
    const actualRevenue = data.actual_revenue;
    const roi = data.roi;
    const metrics = data.model_metrics || {};
    
    let html = `
        <div class="result-card" style="padding: 20px; background: #1e2139; border-radius: 8px; margin-top: 20px;">
            <h3 style="color: #fff; margin-bottom: 10px;">${movie.title}</h3>
            <p style="color: #b8bcc8; margin: 5px 0;"><strong>Год:</strong> ${movie.release_date}</p>
            <p style="color: #b8bcc8; margin: 5px 0;"><strong>Жанр:</strong> ${movie.genres}</p>
            <p style="color: #b8bcc8; margin: 5px 0;"><strong>Рейтинг:</strong> ${movie.vote_average.toFixed(1)}/10</p>
            
            <div style="margin-top: 15px; padding: 15px; background: #0f1620; border-radius: 5px;">
                <h4 style="color: #4ECDC4; margin-bottom: 10px;">Финансовые показатели:</h4>
                <p style="color: #b8bcc8;"><strong>Бюджет:</strong> $${budget.toLocaleString()}</p>
                <p style="color: #b8bcc8;"><strong>Предсказанный доход:</strong> $${predictedRevenue.toLocaleString()}</p>
                <p style="color: #b8bcc8;"><strong>Фактический доход:</strong> ${actualRevenue > 0 ? '$' + actualRevenue.toLocaleString() : 'Нет данных'}</p>
                <p style="color: #FFE66D; font-weight: bold;"><strong>ROI (прогноз):</strong> ${roi.toFixed(2)}%</p>
            </div>
    `;
    
    if (Object.keys(metrics).length > 0) {
        html += `
            <div style="margin-top: 15px; padding: 15px; background: #0f1620; border-radius: 5px;">
                <h4 style="color: #FFE66D; margin-bottom: 10px;">Метрики модели регрессии:</h4>
                <p style="color: #b8bcc8;"><strong>R² Score:</strong> ${(metrics.r2 || 0).toFixed(4)}</p>
                <p style="color: #b8bcc8;"><strong>MAE:</strong> $${(metrics.mae || 0).toLocaleString()}</p>
                <p style="color: #b8bcc8;"><strong>RMSE:</strong> $${(metrics.rmse || 0).toLocaleString()}</p>
            </div>
        `;
    }
    
    html += `</div>`;
    document.querySelector('#revenue-section .prediction-result').innerHTML = html;
}

// ============ РЕКОМЕНДАЦИИ ФИЛЬМОВ ============
document.getElementById('recommendations-btn').addEventListener('click', async (e) => {
    e.preventDefault();
    
    const movieTitle = document.getElementById('recommendations-movie-input').value.trim();
    const nRecommendations = parseInt(document.getElementById('n-recommendations').value) || 10;
    
    if (!movieTitle) {
        alert('Пожалуйста, введите название фильма');
        return;
    }
    
    try {
        document.querySelector('#recommendations-section .prediction-result').innerHTML = '<p style="text-align: center;">⏳ Поиск рекомендаций...</p>';
        
        const response = await fetch(`${API_BASE}/recommendations`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                movie_title: movieTitle,
                n_recommendations: nRecommendations
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayRecommendations(data.recommendations);
        } else {
            document.querySelector('#recommendations-section .prediction-result').innerHTML = `<p style="color: red;">❌ Ошибка: ${data.error}</p>`;
        }
    } catch (error) {
        console.error('Ошибка:', error);
        document.querySelector('#recommendations-section .prediction-result').innerHTML = '<p style="color: red;">❌ Ошибка при получении рекомендаций</p>';
    }
});

function displayRecommendations(recommendations) {
    let html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px; margin-top: 20px;">';
    
    recommendations.forEach((movie, index) => {
        html += `
            <div class="result-card" style="padding: 15px; background: #1e2139; border-radius: 8px;">
                <h4 style="color: #4ECDC4; margin-bottom: 10px;">${index + 1}. ${movie.title}</h4>
                <p style="color: #b8bcc8; font-size: 12px;"><strong>Год:</strong> ${movie.release_date}</p>
                <p style="color: #b8bcc8; font-size: 12px;"><strong>Жанр:</strong> ${movie.genres}</p>
                <p style="color: #b8bcc8; font-size: 12px;"><strong>Рейтинг:</strong> ${movie.vote_average.toFixed(1)}/10</p>
                <p style="color: #FFE66D; font-size: 12px; font-weight: bold;"><strong>Схожесть:</strong> ${(movie.similarity_score * 100).toFixed(1)}%</p>
                <p style="color: #b8bcc8; font-size: 11px; line-height: 1.4; margin-top: 10px;">${movie.overview.substring(0, 100)}...</p>
            </div>
        `;
    });
    
    html += '</div>';
    document.querySelector('#recommendations-section .prediction-result').innerHTML = html;
}

// ============ ПОИСК И АВТОДОПОЛНЕНИЕ ============
let debounceTimer;

// Классификация
document.getElementById('movie-input').addEventListener('input', (e) => {
    clearTimeout(debounceTimer);
    const query = e.target.value.trim();
    
    if (query.length < 2) {
        document.getElementById('suggestions').innerHTML = '';
        return;
    }
    
    debounceTimer = setTimeout(() => fetchSuggestions(query, 'suggestions'));
});

// Прогноз доходов
document.getElementById('revenue-movie-input').addEventListener('input', (e) => {
    clearTimeout(debounceTimer);
    const query = e.target.value.trim();
    
    if (query.length < 2) {
        document.getElementById('revenue-suggestions').innerHTML = '';
        return;
    }
    
    debounceTimer = setTimeout(() => fetchSuggestions(query, 'revenue-suggestions'));
});

// Рекомендации
document.getElementById('recommendations-movie-input').addEventListener('input', (e) => {
    clearTimeout(debounceTimer);
    const query = e.target.value.trim();
    
    if (query.length < 2) {
        document.getElementById('recommendations-suggestions').innerHTML = '';
        return;
    }
    
    debounceTimer = setTimeout(() => fetchSuggestions(query, 'recommendations-suggestions'));
});

async function fetchSuggestions(query, elementId) {
    try {
        const response = await fetch(`${API_BASE}/search?q=${encodeURIComponent(query)}`);
        const data = await response.json();
        
        const element = document.getElementById(elementId);
        element.innerHTML = '';
        
        if (data.suggestions && data.suggestions.length > 0) {
            data.suggestions.forEach(suggestion => {
                const li = document.createElement('div');
                li.className = 'suggestion-item';
                li.textContent = suggestion;
                li.style.cssText = 'padding: 10px; cursor: pointer; hover-bg: #4ECDC4;';
                li.addEventListener('click', () => {
                    if (elementId === 'suggestions') {
                        document.getElementById('movie-input').value = suggestion;
                    } else if (elementId === 'revenue-suggestions') {
                        document.getElementById('revenue-movie-input').value = suggestion;
                    } else {
                        document.getElementById('recommendations-movie-input').value = suggestion;
                    }
                    element.innerHTML = '';
                });
                element.appendChild(li);
            });
        }
    } catch (error) {
        console.error('Ошибка поиска:', error);
    }
}

// ============ ИНИЦИАЛИЗАЦИЯ ============
document.addEventListener('DOMContentLoaded', () => {
    console.log('%c🎬 MovieMatch приложение с ансамблевыми методами загружено!', 'color: #4ECDC4; font-size: 16px; font-weight: bold;');
    console.log('%cДоступные модели классификации:', 'color: #FFE66D; font-size: 12px;');
    console.log('%c  - Random Forest', 'color: #FF6B6B; font-size: 12px;');
    console.log('%c  - Gradient Boosting', 'color: #FF6B6B; font-size: 12px;');
    console.log('%c  - Logistic Regression + k-NN', 'color: #FF6B6B; font-size: 12px;');
    console.log('%c  - Decision Tree', 'color: #FF6B6B; font-size: 12px;');
});

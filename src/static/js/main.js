/**
 * MovieRec Unified - Main JavaScript
 * Handles client-side interactions for the recommendation platform
 */

(function() {
    'use strict';

    // API helper functions
    const API = {
        baseUrl: '',

        async get(endpoint, params = {}) {
            const url = new URL(endpoint, window.location.origin);
            Object.entries(params).forEach(([key, value]) => {
                if (value !== undefined && value !== null) {
                    url.searchParams.append(key, value);
                }
            });
            
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`API Error: ${response.status}`);
            }
            return response.json();
        },

        async post(endpoint, data) {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });
            if (!response.ok) {
                throw new Error(`API Error: ${response.status}`);
            }
            return response.json();
        },

        // Recommendation endpoints
        async getRecommendations(userId, topK = 10, model = null) {
            return this.get('/api/recommend', { user_id: userId, top_k: topK, model });
        },

        async getSimilarItems(itemId, topK = 10) {
            return this.get(`/api/similar/${itemId}`, { top_k: topK });
        },

        async getBatchRecommendations(userIds, topK = 10, model = null) {
            return this.post('/api/batch-recommend', { user_ids: userIds, top_k: topK, model });
        },

        async getModels() {
            return this.get('/api/models');
        },

        async getMovieDetails(movieId) {
            return this.get(`/api/movie/${movieId}`);
        },

        async healthCheck() {
            return this.get('/api/health');
        },
    };

    // UI Components
    const UI = {
        // Create a movie card element
        createMovieCard(movie, rank = null) {
            const card = document.createElement('div');
            card.className = 'col';
            
            const genres = Array.isArray(movie.genres) 
                ? movie.genres.slice(0, 3).join(', ')
                : (movie.genres || '');
            
            const posterHtml = movie.poster_path
                ? `<img src="https://image.tmdb.org/t/p/w500${movie.poster_path}" class="card-img-top" alt="${movie.title}">`
                : `<div class="card-img-placeholder d-flex align-items-center justify-content-center"><i class="fas fa-film fa-3x text-muted"></i></div>`;
            
            const rankBadge = rank !== null
                ? `<span class="badge bg-primary position-absolute top-0 start-0 m-2">#${rank}</span>`
                : '';
            
            card.innerHTML = `
                <div class="card h-100 movie-card">
                    <div class="card-img-wrapper position-relative">
                        ${posterHtml}
                        ${rankBadge}
                    </div>
                    <div class="card-body">
                        <h6 class="card-title text-truncate" title="${movie.title}">
                            ${movie.title || 'Unknown Title'}
                        </h6>
                        ${genres ? `<p class="card-text small text-muted text-truncate">${genres}</p>` : ''}
                    </div>
                    <div class="card-footer bg-transparent border-0">
                        <a href="/movie/${movie.item_id}" class="btn btn-sm btn-outline-primary w-100">
                            View Details
                        </a>
                    </div>
                </div>
            `;
            
            return card;
        },

        // Show loading spinner
        showLoading(container) {
            container.innerHTML = `
                <div class="text-center py-5">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-3 text-muted">Loading recommendations...</p>
                </div>
            `;
        },

        // Show error message
        showError(container, message) {
            container.innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    ${message}
                </div>
            `;
        },

        // Show success message
        showSuccess(container, message) {
            container.innerHTML = `
                <div class="alert alert-success">
                    <i class="fas fa-check-circle me-2"></i>
                    ${message}
                </div>
            `;
        },

        // Show info message
        showInfo(container, message) {
            container.innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-info-circle me-2"></i>
                    ${message}
                </div>
            `;
        },
    };

    // Page handlers
    const Pages = {
        // Initialize based on current page
        init() {
            const path = window.location.pathname;
            
            // Common initialization
            this.initNavigation();
            
            // Page-specific initialization
            if (path === '/' || path === '/index') {
                this.initIndexPage();
            } else if (path.startsWith('/home')) {
                this.initHomePage();
            } else if (path.startsWith('/movie/')) {
                this.initMovieDetailsPage();
            } else if (path.startsWith('/recommendations')) {
                this.initRecommendationsPage();
            }
        },

        initNavigation() {
            // Highlight active nav item
            const currentPath = window.location.pathname;
            document.querySelectorAll('.navbar-nav .nav-link').forEach(link => {
                if (link.getAttribute('href') === currentPath) {
                    link.classList.add('active');
                }
            });
        },

        initIndexPage() {
            // Auto-focus on user ID input
            const userInput = document.querySelector('input[name="user_id"]');
            if (userInput) {
                userInput.focus();
            }
        },

        initHomePage() {
            // Could add infinite scroll, filtering, etc.
        },

        initMovieDetailsPage() {
            // Could add rating functionality, etc.
        },

        initRecommendationsPage() {
            // Handle model selection change
            const modelSelect = document.getElementById('model');
            if (modelSelect) {
                modelSelect.addEventListener('change', function() {
                    // Could auto-submit form or show preview
                });
            }
        },
    };

    // Utility functions
    const Utils = {
        // Format date
        formatDate(dateStr) {
            if (!dateStr) return '';
            const date = new Date(dateStr);
            return date.toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'long',
                day: 'numeric',
            });
        },

        // Debounce function
        debounce(func, wait) {
            let timeout;
            return function executedFunction(...args) {
                const later = () => {
                    clearTimeout(timeout);
                    func(...args);
                };
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            };
        },

        // Throttle function
        throttle(func, limit) {
            let inThrottle;
            return function executedFunction(...args) {
                if (!inThrottle) {
                    func(...args);
                    inThrottle = true;
                    setTimeout(() => inThrottle = false, limit);
                }
            };
        },

        // Local storage helpers
        storage: {
            get(key, defaultValue = null) {
                try {
                    const item = localStorage.getItem(key);
                    return item ? JSON.parse(item) : defaultValue;
                } catch (e) {
                    return defaultValue;
                }
            },
            set(key, value) {
                try {
                    localStorage.setItem(key, JSON.stringify(value));
                } catch (e) {
                    console.error('LocalStorage error:', e);
                }
            },
            remove(key) {
                localStorage.removeItem(key);
            },
        },
    };

    // Initialize on DOM ready
    document.addEventListener('DOMContentLoaded', () => {
        Pages.init();
    });

    // Expose to global scope for inline scripts
    window.MovieRec = {
        API,
        UI,
        Utils,
    };

})();

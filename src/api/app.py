"""
Flask application factory and routes.
"""
from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, url_for

from .inference import InferenceService


def create_app(
    config: dict | None = None,
    model_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> Flask:
    """
    Create and configure the Flask application.
    
    Args:
        config: Flask configuration dictionary
        model_dir: Directory containing model checkpoints
        data_dir: Directory containing data files
        
    Returns:
        Configured Flask application
    """
    # Get the directory where this file is located
    current_dir = Path(__file__).parent.parent
    
    app = Flask(
        __name__,
        template_folder=str(current_dir / "templates"),
        static_folder=str(current_dir / "static"),
    )
    
    # Default configuration
    app.config.update({
        "SECRET_KEY": os.environ.get("SECRET_KEY", "dev-secret-key"),
        "MODEL_DIR": model_dir or os.environ.get("MODEL_DIR", "checkpoints"),
        "DATA_DIR": data_dir or os.environ.get("DATA_DIR", "data/movielens-small"),
        "DEFAULT_MODEL": os.environ.get("DEFAULT_MODEL", "lightgcn"),
        "TOP_K": int(os.environ.get("TOP_K", "10")),
    })
    
    # Override with provided config
    if config:
        app.config.update(config)
    
    # Initialize inference service
    inference_service = InferenceService(
        model_dir=app.config["MODEL_DIR"],
        data_dir=app.config["DATA_DIR"],
        device=os.environ.get("DEVICE", "cpu"),
    )
    
    # Store in app context
    app.inference_service = inference_service
    
    # Register routes
    register_routes(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    return app


def register_routes(app: Flask) -> None:
    """Register all routes."""
    
    @app.route("/")
    def index():
        """Home page."""
        return render_template("index.html")
    
    @app.route("/home")
    def home():
        """User home page with recommendations."""
        user_id = request.args.get("user_id", type=int)
        if user_id is None:
            return redirect(url_for("index"))
        
        service: InferenceService = app.inference_service
        
        try:
            recommendations = service.recommend(
                user_id=user_id,
                top_k=app.config["TOP_K"],
            )
        except ValueError as e:
            recommendations = []
        
        return render_template(
            "home.html",
            user_id=user_id,
            recommendations=recommendations,
        )
    
    @app.route("/movie/<int:movie_id>")
    def movie_details(movie_id: int):
        """Movie details page."""
        service: InferenceService = app.inference_service
        
        # Get movie info
        movie = service.movie_details.get(movie_id, {})
        movie["id"] = movie_id
        
        # Get similar movies
        try:
            similar = service.recommend_similar(movie_id, top_k=6)
        except ValueError:
            similar = []
        
        return render_template(
            "movie_details.html",
            movie=movie,
            similar_movies=similar,
        )
    
    @app.route("/recommendations")
    def recommendations():
        """Recommendations page."""
        user_id = request.args.get("user_id", type=int)
        model = request.args.get("model", app.config["DEFAULT_MODEL"])
        top_k = request.args.get("top_k", app.config["TOP_K"], type=int)
        
        if user_id is None:
            return redirect(url_for("index"))
        
        service: InferenceService = app.inference_service
        
        try:
            recs = service.recommend(
                user_id=user_id,
                top_k=top_k,
                model_name=model,
            )
        except ValueError as e:
            recs = []
        
        return render_template(
            "recommendations.html",
            user_id=user_id,
            recommendations=recs,
            model=model,
            available_models=service.registry.list_models(),
        )
    
    @app.route("/top-rated")
    def top_rated():
        """Top rated movies page."""
        return render_template("top_rated_movies.html")
    
    @app.route("/profile")
    def user_profile():
        """User profile page."""
        user_id = request.args.get("user_id", type=int)
        if user_id is None:
            return redirect(url_for("index"))
        
        return render_template(
            "user_profile.html",
            user_id=user_id,
        )
    
    # API Routes
    @app.route("/api/recommend", methods=["GET", "POST"])
    def api_recommend():
        """API endpoint for recommendations."""
        if request.method == "POST":
            data = request.get_json()
            user_id = data.get("user_id")
            top_k = data.get("top_k", app.config["TOP_K"])
            model = data.get("model")
            exclude_items = data.get("exclude_items", [])
        else:
            user_id = request.args.get("user_id", type=int)
            top_k = request.args.get("top_k", app.config["TOP_K"], type=int)
            model = request.args.get("model")
            exclude_items = request.args.getlist("exclude", type=int)
        
        if user_id is None:
            return jsonify({"error": "user_id is required"}), 400
        
        service: InferenceService = app.inference_service
        
        try:
            recommendations = service.recommend(
                user_id=user_id,
                top_k=top_k,
                model_name=model,
                interacted_items=exclude_items,
            )
            return jsonify({
                "user_id": user_id,
                "recommendations": recommendations,
                "model": model or service.registry.default_model,
            })
        except ValueError as e:
            return jsonify({"error": str(e)}), 404
    
    @app.route("/api/similar/<int:item_id>")
    def api_similar(item_id: int):
        """API endpoint for similar items."""
        top_k = request.args.get("top_k", 10, type=int)
        model = request.args.get("model")
        
        service: InferenceService = app.inference_service
        
        try:
            similar = service.recommend_similar(
                item_id=item_id,
                top_k=top_k,
                model_name=model,
            )
            return jsonify({
                "item_id": item_id,
                "similar_items": similar,
            })
        except ValueError as e:
            return jsonify({"error": str(e)}), 404
    
    @app.route("/api/batch-recommend", methods=["POST"])
    def api_batch_recommend():
        """API endpoint for batch recommendations."""
        data = request.get_json()
        user_ids = data.get("user_ids", [])
        top_k = data.get("top_k", app.config["TOP_K"])
        model = data.get("model")
        
        if not user_ids:
            return jsonify({"error": "user_ids is required"}), 400
        
        service: InferenceService = app.inference_service
        
        results = service.batch_recommend(
            user_ids=user_ids,
            top_k=top_k,
            model_name=model,
        )
        
        return jsonify({
            "results": {str(k): v for k, v in results.items()},
            "model": model or service.registry.default_model,
        })
    
    @app.route("/api/models")
    def api_list_models():
        """List available models."""
        service: InferenceService = app.inference_service
        return jsonify({
            "models": service.registry.list_models(),
            "default": service.registry.default_model,
        })
    
    @app.route("/api/movie/<int:movie_id>")
    def api_movie_details(movie_id: int):
        """Get movie details."""
        service: InferenceService = app.inference_service
        
        if movie_id not in service.movie_details:
            return jsonify({"error": "Movie not found"}), 404
        
        movie = {"id": movie_id, **service.movie_details[movie_id]}
        return jsonify(movie)
    
    @app.route("/api/health")
    def health_check():
        """Health check endpoint."""
        service: InferenceService = app.inference_service
        return jsonify({
            "status": "healthy",
            "models_loaded": len(service.registry.models),
            "initialized": service._initialized,
        })


def register_error_handlers(app: Flask) -> None:
    """Register error handlers."""
    
    @app.errorhandler(404)
    def not_found(e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Not found"}), 404
        return render_template("index.html"), 404
    
    @app.errorhandler(500)
    def server_error(e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Internal server error"}), 500
        return render_template("index.html"), 500


# CLI entry point
def main():
    """Run the Flask development server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Movie Recommendation API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--model-dir", help="Model checkpoint directory")
    parser.add_argument("--data-dir", help="Data directory")
    parser.add_argument("--device", default="cpu", help="Device for inference")
    
    args = parser.parse_args()
    
    os.environ["DEVICE"] = args.device
    
    app = create_app(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
    )
    
    # Load default model
    service: InferenceService = app.inference_service
    service.initialize()
    
    # Try to load models from checkpoint directory
    model_dir = Path(app.config["MODEL_DIR"])
    if model_dir.exists():
        for checkpoint in model_dir.glob("*.pt"):
            model_name = checkpoint.stem
            model_type = "lightgcn"  # Default, could be inferred from checkpoint
            if "ngcf" in model_name.lower():
                model_type = "ngcf"
            elif "ncf" in model_name.lower():
                model_type = "ncf"
            
            try:
                service.load_model(
                    name=model_name,
                    model_type=model_type,
                    checkpoint_path=checkpoint,
                )
                print(f"Loaded model: {model_name}")
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
    
    print(f"\n🚀 Starting server at http://{args.host}:{args.port}")
    print(f"📊 Models loaded: {service.registry.list_models()}")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

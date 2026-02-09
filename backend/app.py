from __future__ import annotations

from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS

from analysis_service import analyze_portfolio
from services.stress_service import analyze_with_shock

def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/analyze")
    def analyze():
        payload = request.get_json(silent=True) or {}
        try:
            result = analyze_portfolio(payload)
            return jsonify(result)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception:
            return jsonify({"error": "Internal server error"}), 500
        
    @app.post("/api/analyze_shock")
    def analyze_shock():
        payload = request.get_json(silent=True) or {}
        try:
            result = analyze_with_shock(payload)
            return jsonify(result)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception:
            return jsonify({"error": "Internal server error"}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=True)

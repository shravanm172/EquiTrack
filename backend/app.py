from __future__ import annotations

from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS

from providers.market_data import fetch_price_history
from services.analysis_service import analyze_portfolio
from services.stress_service import analyze_with_shock

def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})
    
    @app.post("/api/holdings/validate")
    def validate_holding():
        payload = request.get_json(silent=True) or {}

        ticker = str(payload.get("ticker", "")).strip().upper()
        requested_date = str(payload.get("buy_date", "")).strip()
        lookahead_days = int(payload.get("lookahead_days", 7))

        if not ticker:
            return jsonify({"valid": False, "reason": "ticker is required"}), 400
        if not requested_date:
            return jsonify({"valid": False, "reason": "buy_date is required"}), 400

        try:
            d0 = datetime.strptime(requested_date, "%Y-%m-%d")
        except ValueError:
            return jsonify({"valid": False, "reason": "buy_date must be YYYY-MM-DD"}), 400

        if lookahead_days < 1:
            lookahead_days = 1

        start = d0.strftime("%Y-%m-%d")
        end = (d0 + timedelta(days=lookahead_days)).strftime("%Y-%m-%d")

        try:
            ph = fetch_price_history([ticker], start=start, end=end)
            prices = ph.prices

            if prices.empty or ticker not in prices.columns:
                return jsonify({
                    "valid": False,
                    "ticker": ticker,
                    "requested_date": requested_date,
                    "reason": "no price data returned in lookahead window",
                }), 200

            s = prices[ticker].dropna()
            if s.empty:
                return jsonify({
                    "valid": False,
                    "ticker": ticker,
                    "requested_date": requested_date,
                    "reason": "no valid prices returned in lookahead window",
                }), 200

            as_of_dt = s.index[0]
            as_of = as_of_dt.strftime("%Y-%m-%d")
            px = float(s.iloc[0])

            note = None
            if as_of != requested_date:
                note = f"Market closed on {requested_date}; used next trading day {as_of} for that {ticker} trade."

            return jsonify({
                "valid": True,
                "ticker": ticker,
                "requested_date": requested_date,
                "as_of": as_of,
                "price": round(px, 6),
                "note": note,
            }), 200
        except Exception:
            return jsonify({
                "valid": False,
                "ticker": ticker,
                "requested_date": requested_date,
                "reason": "provider error",
            }), 200

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

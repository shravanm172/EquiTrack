from __future__ import annotations

from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/analyze")
    def analyze():
        """
        v0.1: stub response so frontend can integrate immediately.
        Later: replace with real market data fetch + scenario + metrics.
        """
        payload = request.get_json(silent=True) or {}

        # Minimal validation for now
        portfolio = payload.get("portfolio", {})
        holdings = portfolio.get("holdings", [])
        date_range = payload.get("date_range", {})

        # Create a tiny fake equity curve (10 days) so charts can render.
        start_value = float(portfolio.get("starting_cash", 100000))
        dates = [
            "2026-02-01",
            "2026-02-02",
            "2026-02-03",
            "2026-02-04",
            "2026-02-05",
            "2026-02-06",
            "2026-02-07",
            "2026-02-08",
            "2026-02-09",
            "2026-02-10",
        ]

        baseline_curve = []
        scenario_curve = []
        for i, d in enumerate(dates):
            baseline_curve.append({"date": d, "value": round(start_value * (1 + 0.002 * i), 2)})
            scenario_curve.append({"date": d, "value": round(start_value * (1 + 0.0012 * i), 2)})

        response = {
            "meta": {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "holdings_count": len(holdings),
                "date_range": date_range,
            },
            "baseline": {
                "equity_curve": baseline_curve,
                "metrics": {
                    "ann_return": 0.12,
                    "vol": 0.18,
                    "max_drawdown": -0.08,
                },
            },
            "scenario": {
                "equity_curve": scenario_curve,
                "metrics": {
                    "ann_return": 0.08,
                    "vol": 0.22,
                    "max_drawdown": -0.11,
                },
            },
            "delta": {
                "metrics": {
                    "ann_return": -0.04,
                    "vol": 0.04,
                    "max_drawdown": -0.03,
                }
            },
        }

        return jsonify(response)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=True)

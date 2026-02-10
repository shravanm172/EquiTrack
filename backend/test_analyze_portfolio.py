from __future__ import annotations

import json
from pprint import pprint

# Adjust this import to your actual module path
# e.g. from services.analyze_service import analyze_portfolio
from services.analysis_service import analyze_portfolio


def run_case(name: str, payload: dict) -> None:
    print("\n" + "=" * 80)
    print(f"CASE: {name}")
    print("=" * 80)
    print("REQUEST PAYLOAD:")
    print(json.dumps(payload, indent=2))

    try:
        resp = analyze_portfolio(payload)
        print("\nRESPONSE (truncated):")
        # Print key parts so output isn't massive
        pprint(
            {
                "inputs": resp.get("inputs"),
                "metrics": resp.get("metrics"),
                "equity_curve_first3": resp.get("equity_curve", [])[:3],
                "equity_curve_last3": resp.get("equity_curve", [])[-3:],
                "holdings_breakdown": resp.get("holdings_breakdown"),
            }
        )
    except Exception as e:
        print("\nERROR:", repr(e))


def main() -> None:
    # --- WEIGHTS MODE TEST ---
    weights_payload = {
        "portfolio": {
            "starting_cash": 100000,
            "holdings": [
                {"ticker": "AAPL", "weight": 0.5},
                {"ticker": "MSFT", "weight": 0.5},
            ],
        },
        "date_range": {"start": "2024-01-02", "end": "2024-12-31"},
    }

    # --- SHARES MODE TEST ---
    shares_payload = {
        "portfolio": {
            # omit starting_cash to test defaulting to market value on start date
            "holdings": [
                {"ticker": "AAPL", "shares": 10},
                {"ticker": "MSFT", "shares": 5},
            ],
        },
        "date_range": {"start": "2024-01-02", "end": "2024-12-31"},
    }

    run_case("Weights Mode", weights_payload)
    run_case("Shares Mode", shares_payload)


if __name__ == "__main__":
    main()

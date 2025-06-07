from flask import Flask, request, jsonify
import pandas as pd
import yfinance as yf
import numpy as np

app = Flask(__name__)


def compute_metrics(prices: pd.Series):
    returns = prices.pct_change().dropna()
    cumulative_return = (1 + returns).prod() - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    return cumulative_return, volatility, sharpe


@app.route("/")
def index():
    return "Portfolio Analysis API"


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    tickers = data.get("tickers", [])
    weights = data.get("weights")
    benchmark = data.get("benchmark", "SPY")

    if not tickers:
        return jsonify({"error": "tickers list required"}), 400

    if weights is None:
        weights = [1 / len(tickers)] * len(tickers)
    elif len(weights) != len(tickers):
        return jsonify({"error": "weights length must match tickers"}), 400

    df = yf.download(tickers + [benchmark], period="1y", progress=False)["Adj Close"]

    portfolio_prices = (df[tickers] * weights).sum(axis=1)
    benchmark_prices = df[benchmark]

    port_metrics = compute_metrics(portfolio_prices)
    bench_metrics = compute_metrics(benchmark_prices)

    result = {
        "portfolio": {
            "cumulative_return": port_metrics[0],
            "volatility": port_metrics[1],
            "sharpe_ratio": port_metrics[2],
        },
        "benchmark": {
            "cumulative_return": bench_metrics[0],
            "volatility": bench_metrics[1],
            "sharpe_ratio": bench_metrics[2],
        },
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)

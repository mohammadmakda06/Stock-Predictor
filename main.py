"""
Stock Market Predictor
=======================
Random Forest and Gradient Boosting models to predict short-term stock
price trend direction, using engineered features from historical price,
volume, RSI, MACD, and moving average data.

Usage:
    python main.py                    # runs on synthetic 10+ year demo data
    python main.py --csv aapl.csv     # runs on your own OHLCV CSV
    python main.py --horizon 5        # predict trend N trading days ahead
"""

import argparse
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from data_utils import load_price_data
from features import engineer_features
from train import (
    chronological_split, tune_random_forest, tune_gradient_boosting,
    evaluate_model, top_feature_importances,
)

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Stock Market Predictor")
    parser.add_argument("--csv", type=str, default=None,
                         help="Path to OHLCV CSV (Date,Open,High,Low,Close,Volume). "
                              "If omitted, uses synthetic 10+ year demo data.")
    parser.add_argument("--horizon", type=int, default=5,
                         help="Trading days ahead to predict trend direction for")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    # 1. Load data
    print("=" * 60)
    print("1. Loading historical price data")
    print("=" * 60)
    df = load_price_data(args.csv)
    print(f"Loaded {len(df)} trading days: {df.index[0].date()} -> {df.index[-1].date()}")
    if args.csv is None:
        print("(No --csv given: using synthetic 10+ year demo data. Pass a real")
        print(" OHLCV CSV with --csv to run on actual historical prices.)")

    # 2. Feature engineering
    print("\n" + "=" * 60)
    print("2. Engineering features (MAs, RSI, MACD, volume, volatility)")
    print("=" * 60)
    feat_df, feature_cols = engineer_features(df, horizon=args.horizon)
    X, y = feat_df[feature_cols], feat_df["target"]
    print(f"{len(feature_cols)} features across {len(X)} samples")
    print(f"Target balance -> Up: {y.mean():.1%} | Down/Flat: {1 - y.mean():.1%}")

    # 3. Chronological train/test split
    X_train, X_test, y_train, y_test = chronological_split(X, y, args.test_size)
    print(f"\nTrain: {len(X_train)} samples | Test: {len(X_test)} samples "
          f"(chronological split, no shuffling)")

    results = []

    # 4. Random Forest
    print("\n" + "=" * 60)
    print("3. Tuning Random Forest (GridSearchCV + TimeSeriesSplit)")
    print("=" * 60)
    rf_model, rf_params = tune_random_forest(X_train, y_train)
    print(f"Best params: {rf_params}")
    rf_metrics, rf_cm, rf_report = evaluate_model("Random Forest", rf_model, X_test, y_test)
    results.append(rf_metrics)
    print(f"\nTest performance:\n{rf_report}")

    # 5. Gradient Boosting
    print("=" * 60)
    print("4. Tuning Gradient Boosting (GridSearchCV + TimeSeriesSplit)")
    print("=" * 60)
    gb_model, gb_params = tune_gradient_boosting(X_train, y_train)
    print(f"Best params: {gb_params}")
    gb_metrics, gb_cm, gb_report = evaluate_model("Gradient Boosting", gb_model, X_test, y_test)
    results.append(gb_metrics)
    print(f"\nTest performance:\n{gb_report}")

    # 6. Summary comparison
    print("=" * 60)
    print("5. Model comparison")
    print("=" * 60)
    results_df = __import__("pandas").DataFrame(results).set_index("model")
    print(results_df.round(3).to_string())

    # 7. Feature importances
    best_model, best_name = (rf_model, "Random Forest") if rf_metrics["f1"] >= gb_metrics["f1"] \
        else (gb_model, "Gradient Boosting")
    top_feats = top_feature_importances(best_model, feature_cols)
    print(f"\nTop features ({best_name}):")
    print(top_feats.round(4).to_string())

    # 8. Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(df.index, df["Close"], color="#2563eb", linewidth=1)
    axes[0, 0].set_title("Historical Close Price")
    axes[0, 0].set_ylabel("Price")

    results_df[["accuracy", "precision", "recall", "f1"]].plot(
        kind="bar", ax=axes[0, 1], rot=0,
        color=["#2563eb", "#16a34a", "#f59e0b", "#dc2626"])
    axes[0, 1].set_title("Model Comparison")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend(loc="lower right", fontsize=8)

    for ax, cm, name in zip([axes[1, 0]], [rf_cm if best_name == "Random Forest" else gb_cm],
                             [best_name]):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Confusion Matrix ({name})")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Down/Flat", "Up"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Down/Flat", "Up"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

    top_feats.sort_values().plot(kind="barh", ax=axes[1, 1], color="#7c3aed")
    axes[1, 1].set_title(f"Top Feature Importances ({best_name})")

    plt.tight_layout()
    out_path = "stock_predictor_results.png"
    plt.savefig(out_path, dpi=130)
    print(f"\nSaved results plot -> {out_path}")


if __name__ == "__main__":
    main()

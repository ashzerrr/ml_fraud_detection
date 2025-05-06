from src.data.load_data import load_datasets
from src.data.preprocessing import preprocess_data
from src.features.feature_engineering import engineer_features
from src.models.train_models import train_models
from src.models.evaluate_models import evaluate_models
from src.visualization.visualization import (
    plot_class_distribution,
    plot_transaction_amount_distribution,
    plot_correlation_heatmap,
    plot_transactions_by_hour,
    plot_feature_importance,
    plot_combined_feature_importance
)
import pandas as pd

def main():
    print("\n--- Loading data ---")
    transactions_df, cards_df, users_df, fraud_labels_df = load_datasets()

    print("\n--- Preprocessing data ---")
    merged_df = preprocess_data(transactions_df, cards_df, users_df, fraud_labels_df)

    print("\n--- Feature Engineering ---")
    merged_df = engineer_features(merged_df)

    print("\n--- Visualizing basic distributions ---")
    plot_class_distribution(merged_df)
    plot_transaction_amount_distribution(merged_df)
    plot_correlation_heatmap(merged_df)
    plot_transactions_by_hour(merged_df)

    print("\n--- Training models ---")
    top_features = [
        "merchant_fraud_rate", "fraud_count", "zip_encoded", "use_chip_encoded",
        "mean_transaction", "transaction_id", "mcc_encoded", "merchant_city_encoded",
        "merchant_state_encoded", "time_since_last_transaction"
    ]
    best_rf, best_xgb, X_test, y_test = train_models(merged_df, top_features)

    print("\n--- Evaluating models ---")
    evaluate_models(best_rf, X_test, y_test, "Random Forest")
    evaluate_models(best_xgb, X_test, y_test, "XGBoost")

    print("\n--- Visualizing feature importances ---")
    rf_importance = pd.DataFrame({
        "Feature": top_features,
        "Importance": best_rf.feature_importances_
    })
    xgb_importance = pd.DataFrame({
        "Feature": top_features,
        "Importance": best_xgb.feature_importances_
    })

    plot_feature_importance(rf_importance.sort_values(by="Importance", ascending=False), "Random Forest")
    plot_feature_importance(xgb_importance.sort_values(by="Importance", ascending=False), "XGBoost")

    combined_importance = pd.concat([
        rf_importance.assign(Model="Random Forest"),
        xgb_importance.assign(Model="XGBoost")
    ])
    plot_combined_feature_importance(combined_importance)

if __name__ == "__main__":
    main()


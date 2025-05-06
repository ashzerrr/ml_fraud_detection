# main.py

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.modeling import train_models
from src.visualization import visualize_model_performance

def main():
    print("\n--- Loading data ---")
    transactions_df, cards_df, users_df, fraud_labels_df = load_data()

    print("\n--- Preprocessing data ---")
    merged_df = preprocess_data(transactions_df, cards_df, users_df, fraud_labels_df)

    print("\n--- Engineering features ---")
    merged_df, top_features = engineer_features(merged_df)

    print("\n--- Training and evaluating models ---")
    results = train_models(merged_df, top_features)

    print("\n--- Visualizing model performance ---")
    visualize_model_performance(results)

if __name__ == "__main__":
    main()

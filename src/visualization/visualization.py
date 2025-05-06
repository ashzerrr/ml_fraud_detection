import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Fraud label", data=df, palette="coolwarm")
    plt.title("Fraud vs. Non-Fraud Transactions")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.show()

def plot_transaction_amount_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df['amount'], bins=50, kde=True)
    plt.title("Distribution of Transaction Amounts")
    plt.xlabel("Amount")
    plt.ylabel("Frequency")
    plt.show()

def plot_correlation_heatmap(df):
    plt.figure(figsize=(28, 20))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

def plot_transactions_by_hour(df):
    plt.figure(figsize=(10, 5))
    sns.countplot(x="hour", data=df, hue="Fraud label", palette="coolwarm")
    plt.title("Transaction Frequency by Hour (Fraud vs. Non-Fraud)")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Transaction Count")
    plt.show()

def plot_feature_importance(importances_df, model_name):
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importances_df)
    plt.title(f"{model_name} - Feature Importance")
    plt.show()

def plot_combined_feature_importance(combined_df):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=combined_df, x='Importance', y='Feature', hue='Model', palette="viridis")
    plt.title("Feature Importance Comparison")
    plt.show()

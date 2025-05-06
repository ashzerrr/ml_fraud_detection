import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_recall_curve, auc, roc_curve, ConfusionMatrixDisplay

def evaluate_models(model, X_test, y_test, label):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    print(f"\n{label} Test Results:")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds))

    roc_auc = roc_auc_score(y_test, probs)
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)

    print(f"{label} ROC-AUC: {roc_auc:.4f}")
    print(f"{label} PR-AUC: {pr_auc:.4f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=f'{label} (AUC={roc_auc:.4f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{label} ROC Curve')
    plt.legend()
    plt.show()

    # PR Curve
    plt.plot(recall, precision, label=f'{label} (PR-AUC={pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{label} Precision-Recall Curve')
    plt.legend()
    plt.show()

    ConfusionMatrixDisplay.from_predictions(y_test, preds, display_labels=["Not Fraud", "Fraud"], cmap='Blues')
    plt.title(f'{label} Confusion Matrix')
    plt.show()

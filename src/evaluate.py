"""
evaluate.py
-----------
Full model evaluation:
- Accuracy, Precision, Recall, F1, AUC-ROC
- Confusion matrix
- Feature importance plot
- ROC curves for all 3 models
- Business churn driver analysis
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)

from config import MODELS_DIR, RESULTS_DIR, TARGET_COLUMN, RISK_HIGH, RISK_MEDIUM
from src.preprocessing import preprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# COMPUTE METRICS
# ─────────────────────────────────────────────
def compute_metrics(model, X, y, model_name='Model'):
    """Compute all classification metrics."""
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        'model'    : model_name,
        'accuracy' : accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall'   : recall_score(y, y_pred),
        'f1'       : f1_score(y, y_pred),
        'auc_roc'  : roc_auc_score(y, y_proba)
    }
    return metrics, y_pred, y_proba


# ─────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot and save confusion matrix."""
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Churn', 'Churn'])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=13)

    path = os.path.join(RESULTS_DIR, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    logger.info(f"  Confusion matrix saved: {path}")


# ─────────────────────────────────────────────
# ROC CURVES (ALL MODELS)
# ─────────────────────────────────────────────
def plot_roc_curves(models_data, X_test, y_test):
    """
    Plot ROC curves for all models on the same chart.

    Args:
        models_data (dict): {model_name: model}
    """
    plt.figure(figsize=(8, 6))
    colors = ['#2196F3', '#4CAF50', '#FF5722']

    for (name, model), color in zip(models_data.items(), colors):
        y_proba         = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _     = roc_curve(y_test, y_proba)
        auc             = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', color=color, lw=2)

    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves — All Models', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)

    path = os.path.join(RESULTS_DIR, 'roc_curves.png')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    logger.info(f"  ROC curves saved: {path}")


# ─────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────
def plot_feature_importance(model, feature_names, model_name='XGBoost', top_n=15):
    """Plot top N most important features."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        logger.warning("Model has no feature_importances_ or coef_")
        return

    feat_df = pd.DataFrame({
        'feature'   : feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feat_df, palette='viridis')
    plt.title(f'Top {top_n} Feature Importances — {model_name}', fontsize=14)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('')
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, 'feature_importance.png')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    logger.info(f"  Feature importance plot saved: {path}")

    return feat_df


# ─────────────────────────────────────────────
# MODEL COMPARISON TABLE
# ─────────────────────────────────────────────
def plot_model_comparison(all_metrics):
    """Bar chart comparing all model metrics."""
    df = pd.DataFrame(all_metrics)
    df = df.set_index('model')

    ax = df[['accuracy', 'precision', 'recall', 'f1', 'auc_roc']].plot(
        kind='bar', figsize=(12, 6), colormap='Set2', edgecolor='white'
    )
    plt.title('Model Performance Comparison', fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('')
    plt.xticks(rotation=15, fontsize=11)
    plt.ylim(0.7, 1.0)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, 'model_comparison.png')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    logger.info(f"  Model comparison chart saved: {path}")


# ─────────────────────────────────────────────
# CHURN RISK DISTRIBUTION
# ─────────────────────────────────────────────
def plot_risk_distribution(y_proba):
    """Plot distribution of churn probabilities."""
    plt.figure(figsize=(9, 5))
    plt.hist(y_proba, bins=50, color='#2196F3', edgecolor='white', alpha=0.8)
    plt.axvline(RISK_HIGH,   color='red',    linestyle='--', lw=2, label=f'High Risk (>{RISK_HIGH})')
    plt.axvline(RISK_MEDIUM, color='orange', linestyle='--', lw=2, label=f'Medium Risk (>{RISK_MEDIUM})')
    plt.title('Churn Probability Distribution', fontsize=14)
    plt.xlabel('Predicted Churn Probability', fontsize=12)
    plt.ylabel('Number of Customers', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, 'churn_risk_distribution.png')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    logger.info(f"  Risk distribution plot saved: {path}")


# ─────────────────────────────────────────────
# FULL EVALUATION
# ─────────────────────────────────────────────
def evaluate(input_path=None):
    """Run full evaluation of all trained models."""
    logger.info("🚀 Starting model evaluation...")

    # Load preprocessed data
    data          = preprocess(input_path=input_path, save=False)
    X_test        = data['X_test']
    y_test        = data['y_test']
    feature_names = data['feature_names']

    # Load all models
    model_names = ['logistic_regression', 'random_forest', 'xgboost']
    models      = {}
    for name in model_names:
        path = os.path.join(MODELS_DIR, f'{name}.joblib')
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            logger.warning(f"  Model not found: {path}")

    if not models:
        raise FileNotFoundError("No trained models found. Run train.py first.")

    # Evaluate each model
    all_metrics = []
    for name, model in models.items():
        metrics, y_pred, y_proba = compute_metrics(model, X_test, y_test, name)
        all_metrics.append(metrics)
        plot_confusion_matrix(y_test, y_pred, name)

        print(f"\n{'='*50}")
        print(f"  {name.upper().replace('_', ' ')}")
        print(f"{'='*50}")
        print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

    # ROC curves comparison
    plot_roc_curves(models, X_test, y_test)

    # Model comparison chart
    plot_model_comparison(all_metrics)

    # Feature importance for XGBoost (best model)
    if 'xgboost' in models:
        feat_df = plot_feature_importance(models['xgboost'], feature_names)
        print("\nTop 10 Churn Drivers:")
        print(feat_df.head(10).to_string(index=False))

        # Risk distribution
        y_proba = models['xgboost'].predict_proba(X_test)[:, 1]
        plot_risk_distribution(y_proba)

        high   = (y_proba >= RISK_HIGH).sum()
        medium = ((y_proba >= RISK_MEDIUM) & (y_proba < RISK_HIGH)).sum()
        low    = (y_proba < RISK_MEDIUM).sum()
        print(f"\nRisk Segmentation (Test Set):")
        print(f"  🔴 High Risk   : {high:,}  ({high/len(y_proba)*100:.1f}%)")
        print(f"  🟠 Medium Risk : {medium:,}  ({medium/len(y_proba)*100:.1f}%)")
        print(f"  🟢 Low Risk    : {low:,}  ({low/len(y_proba)*100:.1f}%)")

    # Summary table
    summary = pd.DataFrame(all_metrics).set_index('model')
    print(f"\n{'='*60}")
    print("  FINAL METRICS SUMMARY (Test Set)")
    print(f"{'='*60}")
    print(summary.to_string())
    print(f"{'='*60}\n")

    logger.info("✅ Evaluation complete! All plots saved to results/")
    return all_metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=None)
    args = parser.parse_args()
    evaluate(input_path=args.input)

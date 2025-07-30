"""
Evaluation utilities for fraud detection models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score
)
import shap
import warnings
warnings.filterwarnings('ignore')

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name (str): Name of the model for reporting
    
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'average_precision': average_precision_score(y_test, y_pred_proba)
    }
    
    # Print results
    print(f"\n=== {model_name} Performance ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    
    return metrics

def plot_confusion_matrix(y_test, y_pred, model_name="Model", save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        model_name (str): Name of the model
        save_path (str): Path to save the plot
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curve(y_test, y_pred_proba, model_name="Model", save_path=None):
    """
    Plot ROC curve.
    
    Args:
        y_test: True labels
        y_pred_proba: Predicted probabilities
        model_name (str): Name of the model
        save_path (str): Path to save the plot
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_precision_recall_curve(y_test, y_pred_proba, model_name="Model", save_path=None):
    """
    Plot precision-recall curve.
    
    Args:
        y_test: True labels
        y_pred_proba: Predicted probabilities
        model_name (str): Name of the model
        save_path (str): Path to save the plot
    """
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_shap_plots(model, X_test, feature_names, model_name="Model", save_path=None):
    """
    Create SHAP plots for model interpretability.
    
    Args:
        model: Trained model
        X_test: Test features
        feature_names (list): List of feature names
        model_name (str): Name of the model
        save_path (str): Path to save the plot
    """
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title(f'SHAP Summary Plot - {model_name}')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return explainer, shap_values

def compare_models(model_results, save_path=None):
    """
    Compare multiple models and create comparison plots.
    
    Args:
        model_results (dict): Dictionary with model names as keys and metrics as values
        save_path (str): Path to save the plot
    """
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(model_results).T
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    axes[0, 0].bar(comparison_df.index, comparison_df['accuracy'])
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # F1-Score comparison
    axes[0, 1].bar(comparison_df.index, comparison_df['f1_score'])
    axes[0, 1].set_title('F1-Score Comparison')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # ROC-AUC comparison
    axes[1, 0].bar(comparison_df.index, comparison_df['roc_auc'])
    axes[1, 0].set_title('ROC-AUC Comparison')
    axes[1, 0].set_ylabel('ROC-AUC')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Precision-Recall comparison
    axes[1, 1].bar(comparison_df.index, comparison_df['average_precision'])
    axes[1, 1].set_title('Average Precision Comparison')
    axes[1, 1].set_ylabel('Average Precision')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparison_df 
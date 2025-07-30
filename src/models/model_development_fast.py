import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

print("=== FRAUD DETECTION - FAST MODEL DEVELOPMENT ===\n")

# Load processed data
print("1. Loading processed data...")
train_data = pd.read_csv('train_data_processed.csv')
test_data = pd.read_csv('test_data_processed.csv')

print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Separate features and target
X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

# Remove any rows with NaN values
print("Cleaning data - removing NaN values...")
train_mask = ~(X_train.isnull().any(axis=1) | y_train.isnull())
test_mask = ~(X_test.isnull().any(axis=1) | y_test.isnull())

X_train = X_train[train_mask]
y_train = y_train[train_mask]
X_test = X_test[test_mask]
y_test = y_test[test_mask]

print(f"After cleaning - Training data shape: {X_train.shape}")
print(f"After cleaning - Test data shape: {X_test.shape}")

print(f"Features: {list(X_train.columns)}")
print(f"Number of features: {len(X_train.columns)}")

# 2. Quick Feature Selection
print("\n2. Quick Feature Selection...")

# Use ANOVA F-test for feature selection
print("Applying ANOVA F-test for feature selection...")
selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

selected_features = X_train.columns[selector.get_support()]
print(f"Selected features: {list(selected_features)}")

# 3. Model Development
print("\n3. Model Development...")

# Define models (focusing on the most effective ones)
models = {
    'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

# Train and evaluate baseline models
print("Training baseline models...")
baseline_results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_selected, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_selected)
    y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
    
    # Metrics
    accuracy = model.score(X_test_selected, y_test)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    baseline_results[name] = {
        'model': model,
        'accuracy': accuracy,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"✓ {name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")

# 4. Hyperparameter Tuning (Simplified)
print("\n4. Hyperparameter Tuning...")

# Focus on the best performing models
best_models = ['Random Forest', 'Gradient Boosting']

# Simplified hyperparameter grids
param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.1, 0.2]
    }
}

tuned_models = {}

for model_name in best_models:
    print(f"Tuning {model_name}...")
    
    # Get the baseline model
    baseline_model = baseline_results[model_name]['model']
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        baseline_model,
        param_grids[model_name],
        cv=3,  # Reduced CV folds for speed
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_selected, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate best model
    y_pred = best_model.predict(X_test_selected)
    y_pred_proba = best_model.predict_proba(X_test_selected)[:, 1]
    
    accuracy = best_model.score(X_test_selected, y_test)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    tuned_models[model_name] = {
        'model': best_model,
        'accuracy': accuracy,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'best_params': grid_search.best_params_
    }
    
    print(f"✓ {model_name} tuned - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    print(f"  Best parameters: {grid_search.best_params_}")

# 5. Model Evaluation and Comparison
print("\n5. Model Evaluation and Comparison...")

# Compare baseline vs tuned models
print("Model Performance Comparison:")
print("-" * 70)
print(f"{'Model':<20} {'Baseline AUC':<15} {'Tuned AUC':<15} {'Improvement':<15}")
print("-" * 70)

for model_name in best_models:
    baseline_auc = baseline_results[model_name]['auc']
    tuned_auc = tuned_models[model_name]['auc']
    improvement = tuned_auc - baseline_auc
    
    print(f"{model_name:<20} {baseline_auc:<15.4f} {tuned_auc:<15.4f} {improvement:<15.4f}")

# Find best model
best_model_name = max(tuned_models.keys(), key=lambda x: tuned_models[x]['auc'])
best_model = tuned_models[best_model_name]['model']
best_predictions = tuned_models[best_model_name]['predictions']
best_probabilities = tuned_models[best_model_name]['probabilities']

print(f"\n🏆 Best Model: {best_model_name} (AUC: {tuned_models[best_model_name]['auc']:.4f})")

# 6. Detailed Evaluation of Best Model
print(f"\n6. Detailed Evaluation of {best_model_name}...")

# Classification Report
print("Classification Report:")
print(classification_report(y_test, best_predictions))

# Confusion Matrix
cm = confusion_matrix(y_test, best_predictions)
print(f"Confusion Matrix:")
print(cm)

# Feature Importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    print("\nFeature Importance:")
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance)

# 7. Visualization
print("\n7. Creating visualizations...")

# ROC Curves
plt.figure(figsize=(10, 6))

# Plot ROC curves for all tuned models
for model_name, results in tuned_models.items():
    fpr, tpr, _ = roc_curve(y_test, results['probabilities'])
    auc = results['auc']
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Model Comparison')
plt.legend()
plt.grid(True)
plt.savefig('roc_curves_fast.png', dpi=300, bbox_inches='tight')
plt.show()

# Precision-Recall Curve for best model
plt.figure(figsize=(8, 6))
precision, recall, _ = precision_recall_curve(y_test, best_probabilities)
plt.plot(recall, precision, label=f'{best_model_name}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve - {best_model_name}')
plt.legend()
plt.grid(True)
plt.savefig('precision_recall_curve_fast.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature Importance Plot (if available)
if hasattr(best_model, 'feature_importances_'):
    plt.figure(figsize=(10, 6))
    feature_importance.plot(x='feature', y='importance', kind='barh')
    plt.title(f'Feature Importance - {best_model_name}')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance_fast.png', dpi=300, bbox_inches='tight')
    plt.show()

# 8. Save Best Model
print("\n8. Saving best model...")
import pickle

# Save the best model
with open('best_fraud_model_fast.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save feature selector
with open('feature_selector_fast.pkl', 'wb') as f:
    pickle.dump(selector, f)

print("✓ Best model saved as 'best_fraud_model_fast.pkl'")
print("✓ Feature selector saved as 'feature_selector_fast.pkl'")

# 9. Summary
print("\n=== MODEL DEVELOPMENT SUMMARY ===")
print(f"Best Model: {best_model_name}")
print(f"Best AUC: {tuned_models[best_model_name]['auc']:.4f}")
print(f"Best Accuracy: {tuned_models[best_model_name]['accuracy']:.4f}")
print(f"Features Used: {len(selected_features)}")
print(f"Training Samples: {len(X_train)}")
print(f"Test Samples: {len(X_test)}")

print("\n=== FAST MODEL DEVELOPMENT COMPLETED ===")
print("✓ Multiple algorithms tested")
print("✓ Feature selection performed")
print("✓ Hyperparameter tuning completed")
print("✓ Model evaluation and comparison done")
print("✓ Best model saved")
print("✓ Visualizations created") 
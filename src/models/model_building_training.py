import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=== TASK 2: MODEL BUILDING AND TRAINING ===\n")

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# 1. Data Preparation
print("1. DATA PREPARATION")
print("-" * 50)

# Load the processed data
print("Loading processed data...")
try:
    # Try to load the preprocessed data first
    train_data = pd.read_csv('train_data_processed.csv')
    test_data = pd.read_csv('test_data_processed.csv')
    print("✓ Loaded preprocessed train/test data")
except FileNotFoundError:
    print("Preprocessed data not found. Loading original data and performing preprocessing...")
    
    # Load original data
    fraud_data = pd.read_csv('Fraud_Data.csv')
    ip_country_data = pd.read_csv('IpAddress_to_Country.csv')
    
    # Basic preprocessing
    fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
    fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
    
    # Feature engineering
    fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
    fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.day_name()
    fraud_data['time_since_signup'] = (fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds() / 3600
    
    # Encode categorical features
    from sklearn.preprocessing import LabelEncoder
    categorical_features = ['source', 'browser', 'sex']
    label_encoders = {}
    
    for col in categorical_features:
        le = LabelEncoder()
        fraud_data[col] = le.fit_transform(fraud_data[col].astype(str))
        label_encoders[col] = le
    
    # Prepare features
    feature_cols = ['age', 'purchase_value', 'source', 'browser', 'sex', 'hour_of_day', 'time_since_signup']
    X = fraud_data[feature_cols]
    y = fraud_data['class']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create DataFrames
    train_data = pd.DataFrame(X_train_scaled, columns=feature_cols)
    train_data['class'] = y_train.values
    
    test_data = pd.DataFrame(X_test_scaled, columns=feature_cols)
    test_data['class'] = y_test.values

# Separate features and target
print("Separating features and target...")
X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Features: {list(X_train.columns)}")

# Analyze class distribution
print(f"\nClass distribution in training set:")
print(f"Non-Fraud (0): {sum(y_train == 0)} ({sum(y_train == 0)/len(y_train)*100:.2f}%)")
print(f"Fraud (1): {sum(y_train == 1)} ({sum(y_train == 1)/len(y_train)*100:.2f}%)")

print(f"\nClass distribution in test set:")
print(f"Non-Fraud (0): {sum(y_test == 0)} ({sum(y_test == 0)/len(y_test)*100:.2f}%)")
print(f"Fraud (1): {sum(y_test == 1)} ({sum(y_test == 1)/len(y_test)*100:.2f}%)")

# 2. Model Selection and Training
print("\n2. MODEL SELECTION AND TRAINING")
print("-" * 50)

# Define the two required models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)
}

# Train and evaluate models
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    
    # Store results
    results[name] = {
        'model': model,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr
    }
    
    print(f"✓ {name} training completed")

# 3. Model Evaluation
print("\n3. MODEL EVALUATION")
print("-" * 50)

# Compare models
print("Model Performance Comparison:")
print("=" * 80)
print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC-ROC':<10} {'AUC-PR':<10}")
print("=" * 80)

for name, result in results.items():
    print(f"{name:<20} {result['accuracy']:<10.4f} {result['precision']:<10.4f} "
          f"{result['recall']:<10.4f} {result['f1_score']:<10.4f} "
          f"{result['auc_roc']:<10.4f} {result['auc_pr']:<10.4f}")

# Find best model based on F1-Score (good for imbalanced data)
best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
best_model = results[best_model_name]['model']
best_predictions = results[best_model_name]['predictions']
best_probabilities = results[best_model_name]['probabilities']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"Justification: Selected based on F1-Score ({results[best_model_name]['f1_score']:.4f}) "
      f"which is optimal for imbalanced datasets as it balances precision and recall.")

# 4. Detailed Evaluation of Best Model
print(f"\n4. DETAILED EVALUATION OF {best_model_name}")
print("-" * 50)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, best_predictions, target_names=['Non-Fraud', 'Fraud']))

# Confusion Matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, best_predictions)
print(cm)

# Detailed confusion matrix interpretation
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix Interpretation:")
print(f"True Negatives (TN): {tn} - Correctly identified non-fraud")
print(f"False Positives (FP): {fp} - Incorrectly flagged as fraud")
print(f"False Negatives (FN): {fn} - Missed fraud cases")
print(f"True Positives (TP): {tp} - Correctly identified fraud")

# Feature Importance (for Random Forest)
if hasattr(best_model, 'feature_importances_'):
    print(f"\nFeature Importance ({best_model_name}):")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance)

# 5. Visualization
print("\n5. CREATING VISUALIZATIONS")
print("-" * 50)

# Create a comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Fraud Detection Model Evaluation', fontsize=16)

# 1. ROC Curves
for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
    auc = result['auc_roc']
    axes[0, 0].plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('ROC Curves')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. Precision-Recall Curves
for name, result in results.items():
    precision, recall, _ = precision_recall_curve(y_test, result['probabilities'])
    auc_pr = result['auc_pr']
    axes[0, 1].plot(recall, precision, label=f'{name} (AUC-PR = {auc_pr:.3f})')

axes[0, 1].set_xlabel('Recall')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_title('Precision-Recall Curves')
axes[0, 1].legend()
axes[0, 1].grid(True)

# 3. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Fraud', 'Fraud'],
            yticklabels=['Non-Fraud', 'Fraud'], ax=axes[0, 2])
axes[0, 2].set_title(f'Confusion Matrix - {best_model_name}')
axes[0, 2].set_xlabel('Predicted')
axes[0, 2].set_ylabel('Actual')

# 4. Model Comparison Bar Chart
metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'auc_pr']
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR']

x = np.arange(len(metrics))
width = 0.35

for i, (name, result) in enumerate(results.items()):
    values = [result[metric] for metric in metrics]
    axes[1, 0].bar(x + i*width, values, width, label=name, alpha=0.8)

axes[1, 0].set_xlabel('Metrics')
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('Model Performance Comparison')
axes[1, 0].set_xticks(x + width/2)
axes[1, 0].set_xticklabels(metric_names, rotation=45)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5. Feature Importance (if available)
if hasattr(best_model, 'feature_importances_'):
    feature_importance.plot(x='feature', y='importance', kind='barh', ax=axes[1, 1])
    axes[1, 1].set_title(f'Feature Importance - {best_model_name}')
    axes[1, 1].set_xlabel('Importance')

# 6. Class Distribution
class_counts = [sum(y_test == 0), sum(y_test == 1)]
axes[1, 2].pie(class_counts, labels=['Non-Fraud', 'Fraud'], autopct='%1.1f%%', 
               colors=['lightgreen', 'lightcoral'])
axes[1, 2].set_title('Test Set Class Distribution')

plt.tight_layout()
plt.savefig('model_evaluation_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Cross-Validation
print("\n6. CROSS-VALIDATION")
print("-" * 50)

print("Performing 5-fold cross-validation...")
cv_results = {}

for name, model in models.items():
    # Cross-validation scores
    cv_accuracy = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    cv_auc = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    
    cv_results[name] = {
        'accuracy_mean': cv_accuracy.mean(),
        'accuracy_std': cv_accuracy.std(),
        'f1_mean': cv_f1.mean(),
        'f1_std': cv_f1.std(),
        'auc_mean': cv_auc.mean(),
        'auc_std': cv_auc.std()
    }
    
    print(f"\n{name} Cross-Validation Results:")
    print(f"  Accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std() * 2:.4f})")
    print(f"  F1-Score: {cv_f1.mean():.4f} (+/- {cv_f1.std() * 2:.4f})")
    print(f"  AUC-ROC: {cv_auc.mean():.4f} (+/- {cv_auc.std() * 2:.4f})")

# 7. Model Persistence
print("\n7. MODEL PERSISTENCE")
print("-" * 50)

import pickle

# Save the best model
with open('best_fraud_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save all results
with open('model_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("✓ Best model saved as 'best_fraud_model.pkl'")
print("✓ All model results saved as 'model_results.pkl'")

# 8. Summary and Recommendations
print("\n8. SUMMARY AND RECOMMENDATIONS")
print("-" * 50)

print("MODEL PERFORMANCE SUMMARY:")
print(f"Best Model: {best_model_name}")
print(f"F1-Score: {results[best_model_name]['f1_score']:.4f}")
print(f"AUC-ROC: {results[best_model_name]['auc_roc']:.4f}")
print(f"AUC-PR: {results[best_model_name]['auc_pr']:.4f}")
print(f"Precision: {results[best_model_name]['precision']:.4f}")
print(f"Recall: {results[best_model_name]['recall']:.4f}")

print(f"\nJUSTIFICATION FOR BEST MODEL:")
print(f"The {best_model_name} was selected as the best model based on the following criteria:")
print(f"1. F1-Score: {results[best_model_name]['f1_score']:.4f} - Optimal for imbalanced datasets")
print(f"2. AUC-PR: {results[best_model_name]['auc_pr']:.4f} - Excellent precision-recall balance")
print(f"3. AUC-ROC: {results[best_model_name]['auc_roc']:.4f} - Strong overall performance")

if best_model_name == 'Random Forest':
    print(f"4. Feature Importance: Provides interpretable feature rankings")
    print(f"5. Robustness: Less sensitive to outliers and noise")

print(f"\nRECOMMENDATIONS:")
print(f"1. Use {best_model_name} for production fraud detection")
print(f"2. Monitor model performance regularly with new data")
print(f"3. Consider ensemble methods for further improvement")
print(f"4. Implement real-time scoring for immediate fraud detection")

print(f"\n=== TASK 2 COMPLETED SUCCESSFULLY ===")
print("✓ Data preparation completed")
print("✓ Train-test split performed")
print("✓ Logistic Regression baseline model trained")
print("✓ Random Forest ensemble model trained")
print("✓ Models evaluated with imbalanced data metrics")
print("✓ Best model identified and justified")
print("✓ Comprehensive visualizations created")
print("✓ Cross-validation performed")
print("✓ Models saved for deployment") 
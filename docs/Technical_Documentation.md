# Technical Documentation: Fraud Detection System

## Architecture Overview

The fraud detection system follows a modular architecture designed for scalability, maintainability, and production deployment.

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Preprocessing  │    │   Model Layer   │
│                 │    │                 │    │                 │
│ • Fraud_Data    │───▶│ • Feature Eng.  │───▶│ • LightGBM      │
│ • IP_Country    │    │ • Data Cleaning │    │ • Random Forest │
│ • Real-time     │    │ • Scaling       │    │ • XGBoost       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Evaluation    │    │   Deployment    │
                       │                 │    │                 │
                       │ • Metrics       │    │ • API Endpoint  │
                       │ • SHAP Analysis │    │ • Batch Scoring │
                       │ • Performance   │    │ • Monitoring    │
                       └─────────────────┘    └─────────────────┘
```

## Data Pipeline

### 1. Data Ingestion

**Input Data Sources**:
- `Fraud_Data.csv`: Main transaction dataset
- `IpAddress_to_Country.csv`: IP geolocation mapping
- Real-time transaction stream (future enhancement)

**Data Validation**:
```python
def validate_data(data):
    """Validate data quality and schema."""
    required_columns = ['user_id', 'signup_time', 'purchase_time', 
                       'purchase_value', 'class']
    
    # Check required columns
    missing_cols = set(required_columns) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check data types
    data['signup_time'] = pd.to_datetime(data['signup_time'])
    data['purchase_time'] = pd.to_datetime(data['purchase_time'])
    
    # Check for missing values
    null_counts = data.isnull().sum()
    if null_counts.any():
        print(f"Warning: Missing values found:\n{null_counts[null_counts > 0]}")
    
    return data
```

### 2. Feature Engineering

**Temporal Features**:
```python
def create_temporal_features(data):
    """Create time-based features."""
    data['hour_of_day'] = data['purchase_time'].dt.hour
    data['day_of_week'] = data['purchase_time'].dt.day_name()
    data['month'] = data['purchase_time'].dt.month
    data['is_weekend'] = data['purchase_time'].dt.weekday >= 5
    data['time_since_signup'] = (
        data['purchase_time'] - data['signup_time']
    ).dt.total_seconds() / 3600
    
    return data
```

**Behavioral Features**:
```python
def create_behavioral_features(data):
    """Create user behavior features."""
    # Transaction frequency
    user_counts = data.groupby('user_id').size().reset_index(name='transaction_count')
    data = data.merge(user_counts, on='user_id', how='left')
    
    # Transaction velocity
    data['transaction_velocity'] = (
        data['transaction_count'] / (data['time_since_signup'] + 1)
    )
    
    return data
```

**Geolocation Features**:
```python
def create_geolocation_features(data, ip_country_data):
    """Create location-based features."""
    data['ip_address_int'] = data['ip_address'].astype(int)
    
    def find_country(ip_int, ip_country_df):
        mask = ((ip_country_df['lower_bound_ip_address'] <= ip_int) & 
                (ip_int <= ip_country_df['upper_bound_ip_address']))
        matches = ip_country_df[mask]
        return matches.iloc[0]['country'] if len(matches) > 0 else 'Unknown'
    
    data['country'] = data['ip_address_int'].apply(
        lambda x: find_country(x, ip_country_data)
    )
    
    return data
```

### 3. Data Preprocessing

**Categorical Encoding**:
```python
def encode_categorical_features(data, categorical_cols, encoders=None):
    """Encode categorical features using LabelEncoder."""
    encoded_data = data.copy()
    
    if encoders is None:
        encoders = {}
        for col in categorical_cols:
            if col in encoded_data.columns:
                le = LabelEncoder()
                encoded_data[col] = le.fit_transform(encoded_data[col].astype(str))
                encoders[col] = le
    
    return encoded_data, encoders
```

**Feature Scaling**:
```python
def scale_features(X_train, X_test, scaler=None):
    """Scale features using StandardScaler."""
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)
    
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
```

## Model Architecture

### 1. Model Selection

**LightGBM Configuration**:
```python
lightgbm_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}
```

**Training Process**:
```python
def train_lightgbm_model(X_train, y_train, X_val, y_val):
    """Train LightGBM model with early stopping."""
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        lightgbm_params,
        train_data,
        valid_sets=[train_data, val_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    return model
```

### 2. Model Evaluation

**Performance Metrics**:
```python
def evaluate_model_performance(model, X_test, y_test):
    """Comprehensive model evaluation."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'average_precision': average_precision_score(y_test, y_pred_proba)
    }
    
    return metrics, y_pred, y_pred_proba
```

**SHAP Analysis**:
```python
def create_shap_analysis(model, X_test, feature_names):
    """Create SHAP plots for model interpretability."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)
    plt.title('SHAP Summary Plot')
    plt.tight_layout()
    plt.savefig('results/shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return explainer, shap_values
```

## Production Deployment

### 1. Model Serialization

```python
def save_model_artifacts(model, scaler, encoders, model_path='models/'):
    """Save all model artifacts for deployment."""
    # Save model
    joblib.dump(model, f'{model_path}/lightgbm_model.pkl')
    
    # Save preprocessors
    joblib.dump(scaler, f'{model_path}/scaler.pkl')
    joblib.dump(encoders, f'{model_path}/label_encoders.pkl')
    
    # Save feature names
    feature_names = X_train.columns.tolist()
    joblib.dump(feature_names, f'{model_path}/feature_names.pkl')
```

### 2. Prediction Pipeline

```python
class FraudDetectionModel:
    """Production-ready fraud detection model."""
    
    def __init__(self, model_path='models/'):
        self.model = joblib.load(f'{model_path}/lightgbm_model.pkl')
        self.scaler = joblib.load(f'{model_path}/scaler.pkl')
        self.encoders = joblib.load(f'{model_path}/label_encoders.pkl')
        self.feature_names = joblib.load(f'{model_path}/feature_names.pkl')
    
    def preprocess_transaction(self, transaction_data):
        """Preprocess single transaction."""
        # Create features
        features = self.create_features(transaction_data)
        
        # Encode categorical features
        features_encoded, _ = encode_categorical_features(
            features, 
            categorical_cols=['source', 'browser', 'sex', 'country'],
            encoders=self.encoders
        )
        
        # Scale features
        features_scaled = self.scaler.transform(features_encoded)
        
        return features_scaled
    
    def predict(self, transaction_data):
        """Predict fraud probability."""
        features = self.preprocess_transaction(transaction_data)
        fraud_probability = self.model.predict_proba(features)[0][1]
        
        return {
            'fraud_probability': fraud_probability,
            'risk_level': self.get_risk_level(fraud_probability),
            'recommendation': self.get_recommendation(fraud_probability)
        }
    
    def get_risk_level(self, probability):
        """Convert probability to risk level."""
        if probability >= 0.8:
            return 'HIGH'
        elif probability >= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def get_recommendation(self, probability):
        """Get action recommendation."""
        if probability >= 0.8:
            return 'BLOCK'
        elif probability >= 0.4:
            return 'REVIEW'
        else:
            return 'APPROVE'
```

### 3. API Endpoint

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
fraud_model = FraudDetectionModel()

@app.route('/predict', methods=['POST'])
def predict_fraud():
    """API endpoint for fraud prediction."""
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['user_id', 'purchase_value', 'source', 'browser']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Make prediction
        result = fraud_model.predict(data)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

## Monitoring and Maintenance

### 1. Performance Monitoring

```python
def monitor_model_performance(predictions, actual_labels):
    """Monitor model performance metrics."""
    metrics = {
        'accuracy': accuracy_score(actual_labels, predictions),
        'precision': precision_score(actual_labels, predictions),
        'recall': recall_score(actual_labels, predictions),
        'f1_score': f1_score(actual_labels, predictions)
    }
    
    # Log metrics
    logging.info(f"Model performance: {metrics}")
    
    # Alert if performance drops
    if metrics['f1_score'] < 0.8:
        send_alert("Model performance below threshold")
    
    return metrics
```

### 2. Data Drift Detection

```python
def detect_data_dift(current_data, reference_data):
    """Detect data drift using statistical tests."""
    drift_detected = {}
    
    for column in current_data.columns:
        if current_data[column].dtype in ['int64', 'float64']:
            # KS test for numerical features
            statistic, p_value = ks_2samp(
                reference_data[column], 
                current_data[column]
            )
            drift_detected[column] = p_value < 0.05
    
    return drift_detected
```

### 3. Model Retraining

```python
def retrain_model_schedule():
    """Schedule model retraining."""
    # Retrain weekly with new data
    schedule.every().monday.at("02:00").do(retrain_model)
    
    while True:
        schedule.run_pending()
        time.sleep(3600)  # Check every hour

def retrain_model():
    """Retrain model with new data."""
    # Load new data
    new_data = load_new_transactions()
    
    # Retrain model
    model = train_lightgbm_model(new_data)
    
    # Validate performance
    if validate_model_performance(model):
        # Deploy new model
        deploy_model(model)
        logging.info("Model retrained and deployed successfully")
    else:
        logging.error("New model performance below threshold")
```

## Testing

### 1. Unit Tests

```python
import unittest

class TestFraudDetectionModel(unittest.TestCase):
    
    def setUp(self):
        self.model = FraudDetectionModel()
    
    def test_preprocessing(self):
        """Test data preprocessing."""
        test_data = {
            'user_id': 12345,
            'purchase_value': 100.0,
            'source': 'direct',
            'browser': 'chrome'
        }
        
        features = self.model.preprocess_transaction(test_data)
        self.assertEqual(features.shape[1], len(self.model.feature_names))
    
    def test_prediction(self):
        """Test prediction functionality."""
        test_data = {
            'user_id': 12345,
            'purchase_value': 100.0,
            'source': 'direct',
            'browser': 'chrome'
        }
        
        result = self.model.predict(test_data)
        self.assertIn('fraud_probability', result)
        self.assertIn('risk_level', result)
        self.assertIn('recommendation', result)
        self.assertTrue(0 <= result['fraud_probability'] <= 1)

if __name__ == '__main__':
    unittest.main()
```

### 2. Integration Tests

```python
def test_end_to_end_pipeline():
    """Test complete pipeline from data to prediction."""
    # Load test data
    test_data = pd.read_csv('data/test_data_processed.csv')
    
    # Make predictions
    model = FraudDetectionModel()
    predictions = []
    
    for _, row in test_data.iterrows():
        pred = model.predict(row)
        predictions.append(pred['fraud_probability'])
    
    # Evaluate performance
    actual = test_data['class'].values
    predicted = [1 if p > 0.5 else 0 for p in predictions]
    
    f1 = f1_score(actual, predicted)
    assert f1 > 0.8, f"F1 score {f1} below threshold"
```

## Security Considerations

### 1. Input Validation

```python
def validate_input_data(data):
    """Validate and sanitize input data."""
    # Check for SQL injection
    if any(';' in str(v) for v in data.values()):
        raise ValueError("Invalid input detected")
    
    # Check for XSS
    if any('<script>' in str(v).lower() for v in data.values()):
        raise ValueError("Invalid input detected")
    
    # Validate data types
    if not isinstance(data.get('purchase_value'), (int, float)):
        raise ValueError("Invalid purchase value")
    
    return data
```

### 2. Model Security

```python
def secure_model_prediction(model, data):
    """Secure model prediction with input validation."""
    # Validate input
    validated_data = validate_input_data(data)
    
    # Rate limiting
    if not check_rate_limit():
        raise Exception("Rate limit exceeded")
    
    # Make prediction
    result = model.predict(validated_data)
    
    # Log prediction (without sensitive data)
    log_prediction(result['fraud_probability'], result['risk_level'])
    
    return result
```

## Performance Optimization

### 1. Batch Processing

```python
def batch_predict(model, transactions, batch_size=1000):
    """Process predictions in batches for efficiency."""
    predictions = []
    
    for i in range(0, len(transactions), batch_size):
        batch = transactions[i:i + batch_size]
        batch_predictions = model.predict_batch(batch)
        predictions.extend(batch_predictions)
    
    return predictions
```

### 2. Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_country_lookup(ip_address):
    """Cache country lookups for performance."""
    return find_country(ip_address, ip_country_data)
```

This technical documentation provides comprehensive guidance for developers and data scientists working with the fraud detection system, covering all aspects from development to production deployment. 
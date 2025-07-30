# API Documentation: Fraud Detection System

## Overview

The Fraud Detection API provides real-time fraud prediction capabilities for e-commerce transactions. The API accepts transaction data and returns fraud probability scores along with risk assessments and recommendations.

## Base URL

```
Production: https://api.fraud-detection.com/v1
Development: http://localhost:5000/v1
```

## Authentication

All API requests require authentication using API keys.

```bash
Authorization: Bearer YOUR_API_KEY
```

## Endpoints

### 1. Predict Fraud

**Endpoint**: `POST /predict`

**Description**: Predict fraud probability for a single transaction.

**Request Body**:
```json
{
  "user_id": 12345,
  "age": 28,
  "sex": "M",
  "purchase_value": 150.00,
  "source": "direct",
  "browser": "chrome",
  "signup_time": "2024-01-15T10:30:00Z",
  "purchase_time": "2024-01-15T14:45:00Z",
  "ip_address": "192.168.1.1"
}
```

**Response**:
```json
{
  "fraud_probability": 0.23,
  "risk_level": "LOW",
  "recommendation": "APPROVE",
  "confidence": 0.89,
  "features_importance": {
    "transaction_velocity": 0.15,
    "purchase_value": 0.12,
    "time_since_signup": 0.08
  },
  "timestamp": "2024-01-15T14:45:00Z"
}
```

**Status Codes**:
- `200 OK`: Successful prediction
- `400 Bad Request`: Invalid input data
- `401 Unauthorized`: Invalid API key
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

### 2. Batch Predict

**Endpoint**: `POST /predict/batch`

**Description**: Predict fraud probability for multiple transactions.

**Request Body**:
```json
{
  "transactions": [
    {
      "user_id": 12345,
      "age": 28,
      "sex": "M",
      "purchase_value": 150.00,
      "source": "direct",
      "browser": "chrome",
      "signup_time": "2024-01-15T10:30:00Z",
      "purchase_time": "2024-01-15T14:45:00Z",
      "ip_address": "192.168.1.1"
    },
    {
      "user_id": 12346,
      "age": 35,
      "sex": "F",
      "purchase_value": 75.00,
      "source": "organic",
      "browser": "firefox",
      "signup_time": "2024-01-14T09:15:00Z",
      "purchase_time": "2024-01-15T16:20:00Z",
      "ip_address": "192.168.1.2"
    }
  ]
}
```

**Response**:
```json
{
  "predictions": [
    {
      "transaction_id": 0,
      "fraud_probability": 0.23,
      "risk_level": "LOW",
      "recommendation": "APPROVE"
    },
    {
      "transaction_id": 1,
      "fraud_probability": 0.67,
      "risk_level": "MEDIUM",
      "recommendation": "REVIEW"
    }
  ],
  "batch_metrics": {
    "total_transactions": 2,
    "high_risk_count": 0,
    "medium_risk_count": 1,
    "low_risk_count": 1
  },
  "timestamp": "2024-01-15T14:45:00Z"
}
```

### 3. Model Health

**Endpoint**: `GET /health`

**Description**: Check model health and performance metrics.

**Response**:
```json
{
  "status": "healthy",
  "model_version": "1.2.0",
  "last_updated": "2024-01-15T10:00:00Z",
  "performance_metrics": {
    "accuracy": 0.951,
    "precision": 0.876,
    "recall": 0.851,
    "f1_score": 0.863,
    "roc_auc": 0.951
  },
  "system_metrics": {
    "response_time_avg": 45.2,
    "requests_per_minute": 1200,
    "error_rate": 0.001
  }
}
```

### 4. Model Explainability

**Endpoint**: `POST /explain`

**Description**: Get detailed explanation for a prediction using SHAP values.

**Request Body**:
```json
{
  "user_id": 12345,
  "age": 28,
  "sex": "M",
  "purchase_value": 150.00,
  "source": "direct",
  "browser": "chrome",
  "signup_time": "2024-01-15T10:30:00Z",
  "purchase_time": "2024-01-15T14:45:00Z",
  "ip_address": "192.168.1.1"
}
```

**Response**:
```json
{
  "fraud_probability": 0.23,
  "base_value": 0.054,
  "feature_contributions": [
    {
      "feature": "transaction_velocity",
      "value": 0.15,
      "contribution": 0.08,
      "direction": "positive"
    },
    {
      "feature": "purchase_value",
      "value": 150.00,
      "contribution": 0.05,
      "direction": "positive"
    },
    {
      "feature": "time_since_signup",
      "value": 4.25,
      "contribution": -0.02,
      "direction": "negative"
    }
  ],
  "explanation": "This transaction has a low fraud probability due to normal transaction velocity and reasonable time since signup."
}
```

## Data Types

### Transaction Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| user_id | integer | Yes | Unique user identifier |
| age | integer | Yes | User age (18-100) |
| sex | string | Yes | User gender ("M" or "F") |
| purchase_value | float | Yes | Transaction amount (> 0) |
| source | string | Yes | Traffic source ("direct", "organic", "paid") |
| browser | string | Yes | Browser type ("chrome", "firefox", "safari", "edge") |
| signup_time | string | Yes | User signup timestamp (ISO 8601) |
| purchase_time | string | Yes | Transaction timestamp (ISO 8601) |
| ip_address | string | Yes | IP address (IPv4 format) |

### Response Object

| Field | Type | Description |
|-------|------|-------------|
| fraud_probability | float | Probability of fraud (0-1) |
| risk_level | string | Risk category ("LOW", "MEDIUM", "HIGH") |
| recommendation | string | Action recommendation ("APPROVE", "REVIEW", "BLOCK") |
| confidence | float | Model confidence in prediction (0-1) |
| features_importance | object | Top contributing features |
| timestamp | string | Prediction timestamp (ISO 8601) |

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid purchase value",
    "details": {
      "field": "purchase_value",
      "value": -50.00,
      "constraint": "Must be greater than 0"
    }
  },
  "timestamp": "2024-01-15T14:45:00Z"
}
```

### Error Codes

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Invalid input data |
| `AUTHENTICATION_ERROR` | Invalid or missing API key |
| `RATE_LIMIT_ERROR` | Too many requests |
| `MODEL_ERROR` | Model prediction failed |
| `SYSTEM_ERROR` | Internal server error |

## Rate Limiting

- **Free Tier**: 100 requests per hour
- **Professional Tier**: 10,000 requests per hour
- **Enterprise Tier**: 100,000 requests per hour

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 10000
X-RateLimit-Remaining: 9995
X-RateLimit-Reset: 1642248000
```

## SDK Examples

### Python SDK

```python
from fraud_detection import FraudDetectionClient

# Initialize client
client = FraudDetectionClient(api_key="YOUR_API_KEY")

# Single prediction
transaction = {
    "user_id": 12345,
    "age": 28,
    "sex": "M",
    "purchase_value": 150.00,
    "source": "direct",
    "browser": "chrome",
    "signup_time": "2024-01-15T10:30:00Z",
    "purchase_time": "2024-01-15T14:45:00Z",
    "ip_address": "192.168.1.1"
}

result = client.predict(transaction)
print(f"Fraud probability: {result['fraud_probability']}")
print(f"Risk level: {result['risk_level']}")
print(f"Recommendation: {result['recommendation']}")

# Batch prediction
transactions = [transaction1, transaction2, transaction3]
results = client.predict_batch(transactions)

# Get explanation
explanation = client.explain(transaction)
```

### JavaScript SDK

```javascript
const FraudDetection = require('fraud-detection-sdk');

// Initialize client
const client = new FraudDetection('YOUR_API_KEY');

// Single prediction
const transaction = {
    user_id: 12345,
    age: 28,
    sex: 'M',
    purchase_value: 150.00,
    source: 'direct',
    browser: 'chrome',
    signup_time: '2024-01-15T10:30:00Z',
    purchase_time: '2024-01-15T14:45:00Z',
    ip_address: '192.168.1.1'
};

client.predict(transaction)
    .then(result => {
        console.log(`Fraud probability: ${result.fraud_probability}`);
        console.log(`Risk level: ${result.risk_level}`);
        console.log(`Recommendation: ${result.recommendation}`);
    })
    .catch(error => {
        console.error('Prediction failed:', error);
    });
```

## Webhooks

Configure webhooks to receive real-time notifications for high-risk transactions.

### Webhook Configuration

```json
{
  "url": "https://your-app.com/webhooks/fraud",
  "events": ["high_risk_transaction", "model_update"],
  "secret": "your_webhook_secret"
}
```

### Webhook Payload

```json
{
  "event": "high_risk_transaction",
  "timestamp": "2024-01-15T14:45:00Z",
  "data": {
    "transaction_id": "txn_123456",
    "user_id": 12345,
    "fraud_probability": 0.89,
    "risk_level": "HIGH",
    "recommendation": "BLOCK"
  }
}
```

## Best Practices

### 1. Input Validation

- Validate all required fields before sending requests
- Ensure data types match expected formats
- Check for reasonable value ranges

### 2. Error Handling

- Implement proper error handling for all API calls
- Retry failed requests with exponential backoff
- Log errors for debugging and monitoring

### 3. Performance Optimization

- Use batch predictions for multiple transactions
- Implement caching for repeated requests
- Monitor API response times

### 4. Security

- Keep API keys secure and rotate regularly
- Use HTTPS for all API communications
- Validate webhook signatures

## Support

For technical support and questions:

- **Email**: support@fraud-detection.com
- **Documentation**: https://docs.fraud-detection.com
- **Status Page**: https://status.fraud-detection.com
- **GitHub**: https://github.com/fraud-detection/api-examples 
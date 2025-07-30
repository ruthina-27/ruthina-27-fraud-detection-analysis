# Fraud Detection System

## Project Overview

This project implements a comprehensive fraud detection system using machine learning techniques to identify fraudulent transactions in e-commerce data. The system analyzes various features including user behavior, transaction patterns, geolocation data, and temporal characteristics to detect potential fraud.

### Key Features

- **Data Analysis & Preprocessing**: Comprehensive EDA with univariate and bivariate analysis
- **Feature Engineering**: Advanced feature creation including temporal, behavioral, and geolocation features
- **Model Development**: Multiple ML models with performance comparison
- **SHAP Analysis**: Explainable AI using SHAP plots for model interpretability
- **Production Ready**: Complete pipeline from data preprocessing to model deployment

### Problem Statement

E-commerce platforms face significant challenges from fraudulent transactions, which can result in financial losses and damage to customer trust. This project addresses this challenge by:

1. Analyzing transaction patterns and user behavior
2. Identifying key fraud indicators through feature engineering
3. Building robust machine learning models for fraud detection
4. Providing explainable AI insights for model decisions

## Repository Structure

```
fraud-detection/
├── data/                   # Data files (not included in repo due to size)
├── src/                    # Source code
│   ├── preprocessing/      # Data preprocessing scripts
│   ├── analysis/          # Data analysis and visualization
│   ├── models/            # Model training and evaluation
│   └── utils/             # Utility functions
├── models/                # Trained model files
├── results/               # Analysis results and visualizations
├── docs/                  # Documentation
└── requirements.txt       # Python dependencies
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fraud-detection.git
   cd fraud-detection
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv fraud_detection_env
   
   # On Windows
   fraud_detection_env\Scripts\activate
   
   # On macOS/Linux
   source fraud_detection_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare data files**
   - Place your data files in the `data/` directory:
     - `Fraud_Data.csv` - Main transaction data
     - `IpAddress_to_Country.csv` - IP to country mapping

### Running the Project

1. **Data Analysis and Preprocessing**
   ```bash
   python src/analysis/data_analysis_preprocessing.py
   ```

2. **Complete Preprocessing Pipeline**
   ```bash
   python src/preprocessing/complete_preprocessing.py
   ```

3. **Model Building and Training**
   ```bash
   python src/models/model_building_training.py
   ```

4. **Fast Model Development (Alternative)**
   ```bash
   python src/models/model_development_fast.py
   ```

## Model Performance

The system evaluates multiple machine learning models:

- **Logistic Regression**: Baseline model for comparison
- **Random Forest**: Robust ensemble method
- **XGBoost**: Advanced gradient boosting
- **LightGBM**: Fast gradient boosting framework

### Key Metrics

- **AUC-ROC**: Area Under the Receiver Operating Characteristic curve
- **Precision**: Accuracy of positive predictions
- **Recall**: Ability to find all positive cases
- **F1-Score**: Harmonic mean of precision and recall

## Results and Visualizations

The project generates comprehensive visualizations:

- **Univariate Analysis**: Distribution of individual features
- **Bivariate Analysis**: Feature relationships and correlations
- **SHAP Plots**: Model interpretability and feature importance
- **Performance Metrics**: Model comparison charts

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please open an issue in the GitHub repository.

## Acknowledgments

- Dataset providers for the fraud detection data
- Open source community for the machine learning libraries
- Academic research in fraud detection and machine learning 
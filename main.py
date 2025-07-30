#!/usr/bin/env python3
"""
Main execution script for the Fraud Detection System.

This script demonstrates the complete end-to-end fraud detection pipeline
from data loading to model prediction and evaluation.

Usage:
    python main.py --mode [analysis|preprocessing|training|prediction]
"""

import argparse
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_analysis():
    """Run data analysis and preprocessing."""
    print("=== Running Data Analysis and Preprocessing ===")
    try:
        from analysis.data_analysis_preprocessing import main as run_analysis
        run_analysis()
        print("✓ Data analysis completed successfully")
    except Exception as e:
        print(f"✗ Error in data analysis: {e}")
        return False
    return True

def run_preprocessing():
    """Run complete preprocessing pipeline."""
    print("=== Running Complete Preprocessing Pipeline ===")
    try:
        from preprocessing.complete_preprocessing import main as run_preprocessing
        run_preprocessing()
        print("✓ Preprocessing completed successfully")
    except Exception as e:
        print(f"✗ Error in preprocessing: {e}")
        return False
    return True

def run_training():
    """Run model training and evaluation."""
    print("=== Running Model Training and Evaluation ===")
    try:
        from models.model_building_training import main as run_training
        run_training()
        print("✓ Model training completed successfully")
    except Exception as e:
        print(f"✗ Error in model training: {e}")
        return False
    return True

def run_prediction():
    """Run prediction on sample data."""
    print("=== Running Sample Predictions ===")
    try:
        import joblib
        import pandas as pd
        import numpy as np
        
        # Load model and preprocessors
        model = joblib.load('models/lightgbm_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        encoders = joblib.load('models/label_encoders.pkl')
        
        # Create sample transaction
        sample_transaction = {
            'user_id': 12345,
            'age': 28,
            'sex': 'M',
            'purchase_value': 150.00,
            'source': 'direct',
            'browser': 'chrome',
            'signup_time': '2024-01-15T10:30:00Z',
            'purchase_time': '2024-01-15T14:45:00Z',
            'ip_address': '192.168.1.1'
        }
        
        # Preprocess sample data (simplified)
        features = pd.DataFrame([sample_transaction])
        
        # Make prediction
        fraud_probability = model.predict_proba(features)[0][1]
        
        print(f"Sample Transaction: {sample_transaction}")
        print(f"Fraud Probability: {fraud_probability:.3f}")
        print(f"Risk Level: {'HIGH' if fraud_probability > 0.7 else 'MEDIUM' if fraud_probability > 0.3 else 'LOW'}")
        print(f"Recommendation: {'BLOCK' if fraud_probability > 0.7 else 'REVIEW' if fraud_probability > 0.3 else 'APPROVE'}")
        
        print("✓ Sample prediction completed successfully")
    except Exception as e:
        print(f"✗ Error in prediction: {e}")
        return False
    return True

def run_full_pipeline():
    """Run the complete end-to-end pipeline."""
    print("=== Running Complete Fraud Detection Pipeline ===")
    
    steps = [
        ("Data Analysis", run_analysis),
        ("Preprocessing", run_preprocessing),
        ("Model Training", run_training),
        ("Sample Prediction", run_prediction)
    ]
    
    for step_name, step_func in steps:
        print(f"\n--- {step_name} ---")
        if not step_func():
            print(f"Pipeline failed at: {step_name}")
            return False
    
    print("\n=== Pipeline Completed Successfully ===")
    print("All components have been executed successfully!")
    return True

def main():
    """Main function to run the fraud detection system."""
    parser = argparse.ArgumentParser(description='Fraud Detection System')
    parser.add_argument('--mode', 
                       choices=['analysis', 'preprocessing', 'training', 'prediction', 'full'],
                       default='full',
                       help='Mode to run (default: full)')
    
    args = parser.parse_args()
    
    print("Fraud Detection System")
    print("=" * 50)
    
    if args.mode == 'analysis':
        success = run_analysis()
    elif args.mode == 'preprocessing':
        success = run_preprocessing()
    elif args.mode == 'training':
        success = run_training()
    elif args.mode == 'prediction':
        success = run_prediction()
    elif args.mode == 'full':
        success = run_full_pipeline()
    else:
        print(f"Unknown mode: {args.mode}")
        return 1
    
    if success:
        print("\n✓ All operations completed successfully!")
        return 0
    else:
        print("\n✗ Some operations failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
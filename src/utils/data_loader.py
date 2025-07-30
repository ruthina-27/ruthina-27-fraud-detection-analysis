"""
Data loader utilities for fraud detection project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_fraud_data(fraud_file='data/Fraud_Data.csv', ip_file='data/IpAddress_to_Country.csv'):
    """
    Load fraud detection datasets.
    
    Args:
        fraud_file (str): Path to fraud data file
        ip_file (str): Path to IP to country mapping file
    
    Returns:
        tuple: (fraud_data, ip_country_data)
    """
    try:
        fraud_data = pd.read_csv(fraud_file)
        ip_country_data = pd.read_csv(ip_file)
        print(f"✓ Fraud data loaded: {fraud_data.shape}")
        print(f"✓ IP country data loaded: {ip_country_data.shape}")
        return fraud_data, ip_country_data
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None

def load_processed_data(train_file='data/train_data_processed.csv', 
                       test_file='data/test_data_processed.csv'):
    """
    Load preprocessed train and test data.
    
    Args:
        train_file (str): Path to processed training data
        test_file (str): Path to processed test data
    
    Returns:
        tuple: (train_data, test_data)
    """
    try:
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)
        print(f"✓ Processed training data loaded: {train_data.shape}")
        print(f"✓ Processed test data loaded: {test_data.shape}")
        return train_data, test_data
    except FileNotFoundError as e:
        print(f"Error loading processed data: {e}")
        return None, None

def prepare_features_target(data, target_col='class'):
    """
    Separate features and target from dataset.
    
    Args:
        data (pd.DataFrame): Input dataset
        target_col (str): Name of target column
    
    Returns:
        tuple: (X, y)
    """
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42, stratify=True):
    """
    Split data into training and test sets.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Proportion of test set
        random_state (int): Random seed
        stratify (bool): Whether to stratify split
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if stratify:
        return train_test_split(X, y, test_size=test_size, 
                               random_state=random_state, stratify=y)
    else:
        return train_test_split(X, y, test_size=test_size, 
                               random_state=random_state)

def scale_features(X_train, X_test, scaler=None):
    """
    Scale features using StandardScaler.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        scaler: Pre-fitted scaler (optional)
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)
    
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def encode_categorical_features(data, categorical_cols, encoders=None):
    """
    Encode categorical features using LabelEncoder.
    
    Args:
        data (pd.DataFrame): Input data
        categorical_cols (list): List of categorical column names
        encoders (dict): Pre-fitted encoders (optional)
    
    Returns:
        tuple: (encoded_data, encoders)
    """
    encoded_data = data.copy()
    
    if encoders is None:
        encoders = {}
        for col in categorical_cols:
            if col in encoded_data.columns:
                le = LabelEncoder()
                encoded_data[col] = le.fit_transform(encoded_data[col].astype(str))
                encoders[col] = le
    else:
        for col in categorical_cols:
            if col in encoded_data.columns and col in encoders:
                encoded_data[col] = encoders[col].transform(encoded_data[col].astype(str))
    
    return encoded_data, encoders 
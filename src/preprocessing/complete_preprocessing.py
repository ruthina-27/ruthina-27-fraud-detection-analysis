import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("=== COMPLETING REMAINING PREPROCESSING TASKS ===\n")

# Load the datasets
print("1. Loading datasets...")
fraud_data = pd.read_csv('Fraud_Data.csv')
ip_country_data = pd.read_csv('IpAddress_to_Country.csv')
print(f"✓ Fraud_Data.csv loaded: {fraud_data.shape}")
print(f"✓ IpAddress_to_Country.csv loaded: {ip_country_data.shape}")

# Convert data types
fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
fraud_data['ip_address'] = pd.to_numeric(fraud_data['ip_address'], errors='coerce')

print("\n2. Merging Datasets for Geolocation Analysis...")
print("Converting IP addresses to integer format...")

# Convert IP addresses to integer format
fraud_data['ip_address_int'] = fraud_data['ip_address'].astype(int)

# Function to find country for IP address
def find_country(ip_int, ip_country_df):
    mask = (ip_country_df['lower_bound_ip_address'] <= ip_int) & (ip_int <= ip_country_df['upper_bound_ip_address'])
    matches = ip_country_df[mask]
    if len(matches) > 0:
        return matches.iloc[0]['country']
    return 'Unknown'

# Apply the function to get countries
print("Mapping IP addresses to countries...")
fraud_data['country'] = fraud_data['ip_address_int'].apply(lambda x: find_country(x, ip_country_data))

print(f"✓ Countries mapped. Sample countries: {fraud_data['country'].value_counts().head()}")

print("\n3. Feature Engineering...")

# Transaction frequency and velocity
print("Calculating transaction frequency...")
user_transaction_counts = fraud_data.groupby('user_id').size().reset_index(name='transaction_count')
fraud_data = fraud_data.merge(user_transaction_counts, on='user_id', how='left')

# Calculate transaction velocity (transactions per hour since signup)
fraud_data['time_since_signup_hours'] = (fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds() / 3600
fraud_data['transaction_velocity'] = fraud_data['transaction_count'] / (fraud_data['time_since_signup_hours'] + 1)  # +1 to avoid division by zero

# Time-Based features
print("Creating time-based features...")
fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.day_name()
fraud_data['month'] = fraud_data['purchase_time'].dt.month
fraud_data['day_of_month'] = fraud_data['purchase_time'].dt.day

# Time since signup
fraud_data['time_since_signup'] = (fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds() / 3600  # in hours

# Additional time-based features
fraud_data['purchase_hour_category'] = pd.cut(fraud_data['hour_of_day'], 
                                            bins=[0, 6, 12, 18, 24], 
                                            labels=['Night', 'Morning', 'Afternoon', 'Evening'])

# Weekend vs weekday
fraud_data['is_weekend'] = fraud_data['purchase_time'].dt.weekday >= 5

print("✓ Feature engineering completed")
print(f"New features added: transaction_count, transaction_velocity, hour_of_day, day_of_week, month, day_of_month, time_since_signup, purchase_hour_category, is_weekend")

print("\n4. Data Transformation...")

# Analyze class distribution
print("Analyzing class distribution...")
class_dist = fraud_data['class'].value_counts()
print(f"Original class distribution:")
print(f"Non-Fraud (0): {class_dist[0]} ({class_dist[0]/len(fraud_data)*100:.2f}%)")
print(f"Fraud (1): {class_dist[1]} ({class_dist[1]/len(fraud_data)*100:.2f}%)")

# Prepare features for encoding and scaling
categorical_features = ['source', 'browser', 'sex', 'country', 'day_of_week', 'purchase_hour_category']
numerical_features = ['age', 'purchase_value', 'transaction_count', 'transaction_velocity', 
                     'hour_of_day', 'month', 'day_of_month', 'time_since_signup', 'is_weekend']

# Create a copy for encoding
fraud_data_encoded = fraud_data.copy()

# Encode categorical features
print("Encoding categorical features...")
label_encoders = {}
for col in categorical_features:
    if col in fraud_data_encoded.columns:
        le = LabelEncoder()
        fraud_data_encoded[col] = le.fit_transform(fraud_data_encoded[col].astype(str))
        label_encoders[col] = le
        print(f"✓ Encoded {col}")

# Prepare features for modeling
feature_cols = numerical_features + [col for col in categorical_features if col in fraud_data_encoded.columns]
X = fraud_data_encoded[feature_cols]
y = fraud_data_encoded['class']

print(f"✓ Feature encoding completed. Total features: {len(feature_cols)}")

# Split data into train and test sets
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Handle Class Imbalance on training data only
print("\nHandling class imbalance on training data...")
print("Applying SMOTE for oversampling...")
print("Justification: SMOTE creates synthetic samples of the minority class, helping the model learn better patterns without losing information from the majority class.")

smote = SMOTE(random_state=42, sampling_strategy='auto')
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Training set before SMOTE: {X_train.shape}")
print(f"Training set after SMOTE: {X_train_resampled.shape}")
print(f"Class distribution after SMOTE: {np.bincount(y_train_resampled)}")

# Normalization and Scaling
print("\nApplying normalization and scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

print("✓ StandardScaler applied to training and test sets")

# Create final datasets
final_train = pd.DataFrame(X_train_scaled, columns=feature_cols)
final_train['class'] = y_train_resampled

final_test = pd.DataFrame(X_test_scaled, columns=feature_cols)
final_test['class'] = y_test

# Save processed data
print("\n5. Saving processed data...")
fraud_data.to_csv('fraud_data_with_features.csv', index=False)
final_train.to_csv('train_data_processed.csv', index=False)
final_test.to_csv('test_data_processed.csv', index=False)

# Save label encoders for later use
import pickle
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✓ Processed data saved:")
print("- fraud_data_with_features.csv: Original data with all new features")
print("- train_data_processed.csv: Training data with SMOTE and scaling")
print("- test_data_processed.csv: Test data with scaling")
print("- label_encoders.pkl: Label encoders for categorical features")
print("- scaler.pkl: StandardScaler for numerical features")

# Summary statistics
print(f"\n=== FINAL SUMMARY ===")
print(f"Original dataset: {fraud_data.shape}")
print(f"Features created: {len(feature_cols)}")
print(f"Training samples: {len(final_train)}")
print(f"Test samples: {len(final_test)}")
print(f"Fraud rate in training: {final_train['class'].mean() * 100:.2f}%")
print(f"Fraud rate in test: {final_test['class'].mean() * 100:.2f}%")

print("\n=== PREPROCESSING COMPLETED SUCCESSFULLY ===")
print("✓ Datasets merged for geolocation analysis")
print("✓ Feature engineering completed")
print("✓ Class imbalance handled with SMOTE")
print("✓ Data normalized and scaled")
print("✓ Categorical features encoded")
print("✓ Train/test split performed")
print("✓ All processed data saved") 
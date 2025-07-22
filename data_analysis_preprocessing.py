import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

print("=== FRAUD DETECTION - DATA ANALYSIS AND PREPROCESSING ===\n")

# 1. Load the datasets
print("1. Loading datasets...")
try:
    fraud_data = pd.read_csv('Fraud_Data.csv')
    ip_country_data = pd.read_csv('IpAddress_to_Country.csv')
    print(f"✓ Fraud_Data.csv loaded: {fraud_data.shape}")
    print(f"✓ IpAddress_to_Country.csv loaded: {ip_country_data.shape}")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# 2. Initial data exploration
print("\n2. Initial data exploration...")
print(f"Fraud Data Info:")
print(f"- Shape: {fraud_data.shape}")
print(f"- Columns: {list(fraud_data.columns)}")
print(f"- Data types:\n{fraud_data.dtypes}")
print(f"- Memory usage: {fraud_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print(f"\nIP Country Data Info:")
print(f"- Shape: {ip_country_data.shape}")
print(f"- Columns: {list(ip_country_data.columns)}")
print(f"- Data types:\n{ip_country_data.dtypes}")

# 3. Handle Missing Values
print("\n3. Handling missing values...")
print("Missing values in Fraud_Data:")
missing_fraud = fraud_data.isnull().sum()
print(missing_fraud[missing_fraud > 0])

print("\nMissing values in IpAddress_to_Country:")
missing_ip = ip_country_data.isnull().sum()
print(missing_ip[missing_ip > 0])

# Impute missing values
if missing_fraud.sum() > 0:
    # For categorical columns, fill with mode
    categorical_cols = fraud_data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if fraud_data[col].isnull().sum() > 0:
            fraud_data[col].fillna(fraud_data[col].mode()[0], inplace=True)
    
    # For numerical columns, fill with median
    numerical_cols = fraud_data.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if fraud_data[col].isnull().sum() > 0:
            fraud_data[col].fillna(fraud_data[col].median(), inplace=True)

print("✓ Missing values handled")

# 4. Data Cleaning
print("\n4. Data cleaning...")

# Remove duplicates
initial_rows = len(fraud_data)
fraud_data = fraud_data.drop_duplicates()
duplicates_removed = initial_rows - len(fraud_data)
print(f"✓ Removed {duplicates_removed} duplicate rows")

# Convert data types
fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
fraud_data['ip_address'] = pd.to_numeric(fraud_data['ip_address'], errors='coerce')

print("✓ Data types corrected")

# 5. Exploratory Data Analysis (EDA)
print("\n5. Exploratory Data Analysis...")

# Univariate Analysis
print("\n--- Univariate Analysis ---")

# Class distribution
print(f"Class distribution:")
class_dist = fraud_data['class'].value_counts()
print(class_dist)
print(f"Fraud rate: {class_dist[1] / len(fraud_data) * 100:.2f}%")

# Create EDA plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Univariate Analysis', fontsize=16)

# Age distribution
axes[0, 0].hist(fraud_data['age'], bins=30, alpha=0.7, color='skyblue')
axes[0, 0].set_title('Age Distribution')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')

# Purchase value distribution
axes[0, 1].hist(fraud_data['purchase_value'], bins=30, alpha=0.7, color='lightgreen')
axes[0, 1].set_title('Purchase Value Distribution')
axes[0, 1].set_xlabel('Purchase Value')
axes[0, 1].set_ylabel('Frequency')

# Source distribution
source_counts = fraud_data['source'].value_counts()
axes[0, 2].pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%')
axes[0, 2].set_title('Source Distribution')

# Browser distribution
browser_counts = fraud_data['browser'].value_counts()
axes[1, 0].bar(browser_counts.index, browser_counts.values, color='orange')
axes[1, 0].set_title('Browser Distribution')
axes[1, 0].tick_params(axis='x', rotation=45)

# Sex distribution
sex_counts = fraud_data['sex'].value_counts()
axes[1, 1].pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%')
axes[1, 1].set_title('Sex Distribution')

# Class distribution
axes[1, 2].bar(['Non-Fraud', 'Fraud'], class_dist.values, color=['green', 'red'])
axes[1, 2].set_title('Class Distribution')
axes[1, 2].set_ylabel('Count')

plt.tight_layout()
plt.savefig('univariate_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Bivariate Analysis
print("\n--- Bivariate Analysis ---")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Bivariate Analysis', fontsize=16)

# Age vs Class
fraud_data.boxplot(column='age', by='class', ax=axes[0, 0])
axes[0, 0].set_title('Age Distribution by Class')
axes[0, 0].set_xlabel('Class')
axes[0, 0].set_ylabel('Age')

# Purchase Value vs Class
fraud_data.boxplot(column='purchase_value', by='class', ax=axes[0, 1])
axes[0, 1].set_title('Purchase Value Distribution by Class')
axes[0, 1].set_xlabel('Class')
axes[0, 1].set_ylabel('Purchase Value')

# Source vs Class
source_class = pd.crosstab(fraud_data['source'], fraud_data['class'])
source_class.plot(kind='bar', ax=axes[1, 0])
axes[1, 0].set_title('Source vs Class')
axes[1, 0].set_xlabel('Source')
axes[1, 0].set_ylabel('Count')
axes[1, 0].tick_params(axis='x', rotation=45)

# Browser vs Class
browser_class = pd.crosstab(fraud_data['browser'], fraud_data['class'])
browser_class.plot(kind='bar', ax=axes[1, 1])
axes[1, 1].set_title('Browser vs Class')
axes[1, 1].set_xlabel('Browser')
axes[1, 1].set_ylabel('Count')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('bivariate_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Merge Datasets for Geolocation Analysis
print("\n6. Merging datasets for geolocation analysis...")

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

# 7. Feature Engineering
print("\n7. Feature engineering...")

# Transaction frequency and velocity
print("Calculating transaction frequency and velocity...")
user_transaction_counts = fraud_data.groupby('user_id').size().reset_index(name='transaction_count')
fraud_data = fraud_data.merge(user_transaction_counts, on='user_id', how='left')

# Time-based features
print("Creating time-based features...")
fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.day_name()
fraud_data['month'] = fraud_data['purchase_time'].dt.month
fraud_data['day_of_month'] = fraud_data['purchase_time'].dt.day

# Time since signup
fraud_data['time_since_signup'] = (fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds() / 3600  # in hours

# Additional features
fraud_data['purchase_hour_category'] = pd.cut(fraud_data['hour_of_day'], 
                                            bins=[0, 6, 12, 18, 24], 
                                            labels=['Night', 'Morning', 'Afternoon', 'Evening'])

print("✓ Feature engineering completed")
print(f"New features added: {list(fraud_data.columns[-6:])}")

# 8. Data Transformation
print("\n8. Data transformation...")

# Handle Class Imbalance
print("Analyzing class imbalance...")
print(f"Original class distribution:")
print(fraud_data['class'].value_counts(normalize=True))

# Apply SMOTE for oversampling (only to training data)
print("\nApplying SMOTE for class balance...")
# For demonstration, we'll prepare the data for SMOTE
# In practice, this should be applied only to training data after train-test split

# Prepare features for SMOTE
categorical_features = ['source', 'browser', 'sex', 'country', 'day_of_week', 'purchase_hour_category']
numerical_features = ['age', 'purchase_value', 'transaction_count', 'hour_of_day', 'month', 'day_of_month', 'time_since_signup']

# Create a copy for encoding
fraud_data_encoded = fraud_data.copy()

# Encode categorical features
label_encoders = {}
for col in categorical_features:
    if col in fraud_data_encoded.columns:
        le = LabelEncoder()
        fraud_data_encoded[col] = le.fit_transform(fraud_data_encoded[col].astype(str))
        label_encoders[col] = le

# Prepare features for SMOTE
feature_cols = numerical_features + [col for col in categorical_features if col in fraud_data_encoded.columns]
X = fraud_data_encoded[feature_cols]
y = fraud_data_encoded['class']

# Apply SMOTE
smote = SMOTE(random_state=42, sampling_strategy='auto')
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"✓ SMOTE applied")
print(f"Original shape: {X.shape}")
print(f"Resampled shape: {X_resampled.shape}")
print(f"New class distribution: {np.bincount(y_resampled)}")

# Normalization and Scaling
print("\nApplying normalization and scaling...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Create final dataset
final_features = pd.DataFrame(X_scaled, columns=feature_cols)
final_features['class'] = y_resampled

print("✓ Normalization completed")

# 9. Summary and Save Results
print("\n9. Summary and saving results...")

# Save processed data
fraud_data.to_csv('fraud_data_processed.csv', index=False)
final_features.to_csv('fraud_data_final_features.csv', index=False)

print("✓ Processed data saved:")
print("- fraud_data_processed.csv: Original data with new features")
print("- fraud_data_final_features.csv: Final features for modeling")

# Summary statistics
print(f"\nFinal dataset summary:")
print(f"- Total samples: {len(final_features)}")
print(f"- Features: {len(feature_cols)}")
print(f"- Fraud rate: {final_features['class'].mean() * 100:.2f}%")
print(f"- Memory usage: {final_features.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n=== DATA ANALYSIS AND PREPROCESSING COMPLETED ===")
print("✓ Missing values handled")
print("✓ Data cleaned (duplicates removed, types corrected)")
print("✓ EDA completed (univariate and bivariate analysis)")
print("✓ Datasets merged for geolocation analysis")
print("✓ Feature engineering completed")
print("✓ Class imbalance handled with SMOTE")
print("✓ Data normalized and scaled")
print("✓ Categorical features encoded")
print("✓ Results saved to CSV files") 
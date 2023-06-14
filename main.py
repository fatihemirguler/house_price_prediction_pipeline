import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# Load the train data into a pandas DataFrame
train_data = pd.read_csv('train.csv')  # Replace 'train_data.csv' with your actual file path or URL
train_data=train_data.drop('Id',axis=1)
# Print the first few rows of the dataset
# print(train_data.head())

# Get the dimensions of the dataset
print("Data shape:", train_data.shape)

# Compute summary statistics
#print(train_data.describe())

#print(train_data.dtypes.value_counts())
#43 object, 35 int, 3 float

label_encoder = LabelEncoder()
# Apply label encoding to the categorical features
categorical_features_label = ['OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual',
                        'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

for feature in categorical_features_label:
    train_data[feature] = label_encoder.fit_transform(train_data[feature].astype(str))

numerical_features = [col for col in train_data.columns if train_data[col].dtype in ['int64', 'float64']]

categorical_features_onehot = [col for col in train_data.columns if col not in categorical_features_label and col not in numerical_features]

onehot_encoded_data = pd.get_dummies(train_data[categorical_features_onehot])

encoded_train_data = pd.concat([train_data[categorical_features_label], onehot_encoded_data, train_data[numerical_features]], axis=1)

# missing_values = encoded_train_data.isnull().sum()
# print(missing_values[missing_values > 0])
#
# missing_percentage = (missing_values / len(encoded_train_data)) * 100
# print(missing_percentage[missing_percentage > 0])

mean_imputer = SimpleImputer(strategy='mean')
encoded_train_data['LotFrontage'] = mean_imputer.fit_transform(encoded_train_data[['LotFrontage']])
encoded_train_data['MasVnrArea'] = mean_imputer.fit_transform(encoded_train_data[['MasVnrArea']])
encoded_train_data['GarageYrBlt'] = mean_imputer.fit_transform(encoded_train_data[['GarageYrBlt']])

for col in encoded_train_data.columns:
    if encoded_train_data[col].dtype == 'bool':
        encoded_train_data[col] = encoded_train_data[col].astype(int)

Q1 = encoded_train_data.quantile(0.25)
Q3 = encoded_train_data.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

encoded_train_data = encoded_train_data[(encoded_train_data >= lower_bound) & (encoded_train_data <= upper_bound)]
print(encoded_train_data)

missing_values = encoded_train_data.isnull().sum()
print(missing_values[missing_values > 0])
missing_percentage = (missing_values / len(encoded_train_data)) * 100
print(missing_percentage[missing_percentage > 0])

encoded_train_data = encoded_train_data.fillna(encoded_train_data.mean())
print("AAA",encoded_train_data)

# correlation_matrix = encoded_train_data.corr()
# correlation_with_saleprice = correlation_matrix['SalePrice'].sort_values(ascending=False)
# selected_features = correlation_matrix[correlation_matrix['SalePrice'].abs() > 0.0].index.tolist()
# selected_data = encoded_train_data[selected_features]

X = encoded_train_data.drop('SalePrice', axis=1)
y = encoded_train_data['SalePrice']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print('RMSE:', rmse)
std_dev = train_data['SalePrice'].std()
print("Standard Deviation of 'SalePrice': ", std_dev)

print()#PCA
# pca = PCA(n_components=0.95) # keep 95% of variance
# pca.fit(X)
# X_pca = pca.transform(X)
# print(X_pca)
#
# # Split your data
# X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
#
# # Train your model
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# # Predict on the test data
# y_pred = model.predict(X_test)
#
# # Calculate the RMSE
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print("RMSE: ", rmse)










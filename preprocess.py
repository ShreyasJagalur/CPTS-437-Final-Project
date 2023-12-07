import pandas as pd
import numpy as np
import os

# Define the paths
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, '..', 'data')

# Load data
training_data = pd.read_csv(os.path.join(data_dir, "train.csv"))

# Handle missing data for categorical and numerical columns
categorical_columns = training_data.select_dtypes(include='object')
numerical_attributes = training_data.select_dtypes(exclude='object').drop(['SalePrice', 'Id'], axis=1).copy()

for column in ['LotFrontage', 'GarageYrBlt', 'MasVnrArea']:
    numerical_attributes[column].fillna(numerical_attributes[column].mean(), inplace=True)

columns_to_delete_categorical = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu']
categorical_columns = categorical_columns.drop(columns_to_delete_categorical, axis=1)
categorical_columns = categorical_columns.fillna('None')

# Advanced Feature Engineering: Creating an interaction term
training_data['GrLivArea_TotalBsmtSF'] = training_data['GrLivArea'] * training_data['TotalBsmtSF']

# Log Transformation of the Target Variable
training_data['SalePrice'] = np.log(training_data['SalePrice'])

# Prepare the final data frame
final_data_frame = pd.concat([categorical_columns, numerical_attributes, training_data[['GrLivArea_TotalBsmtSF', 'SalePrice']]], axis=1)
final_data_frame_encoded = pd.get_dummies(final_data_frame)

# Save the preprocessed data to the 'src/' directory
preprocessed_file_path = os.path.join(script_dir, "preprocessed_train.csv")
final_data_frame_encoded.to_csv(preprocessed_file_path, index=False)

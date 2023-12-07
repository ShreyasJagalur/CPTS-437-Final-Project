import pandas as pd
import numpy as np
import joblib
import os

# Define the paths
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, '..', 'data')

# Load the trained model
model_file_path = os.path.join(script_dir, "rf_model.joblib")
model = joblib.load(model_file_path)

# Test data preprocessing
test_data = pd.read_csv(os.path.join(data_dir, "test.csv"))
columns_to_delete_categorical = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu']

test_data_clean = test_data.drop(columns_to_delete_categorical + ['Id'], axis=1)
test_data_clean_numeric = test_data_clean.select_dtypes(exclude=['object'])
test_data_clean_category = test_data_clean.select_dtypes(include=['object'])
test_data_clean_numeric = test_data_clean_numeric.fillna(0)
test_data_clean_category = test_data_clean_category.fillna("None")
test_data_clean['GrLivArea_TotalBsmtSF'] = test_data_clean['GrLivArea'] * test_data_clean['TotalBsmtSF']

test_data_final = pd.concat([test_data_clean_category, test_data_clean_numeric, test_data_clean[['GrLivArea_TotalBsmtSF']]], axis=1)
test_data_final_encoded = pd.get_dummies(test_data_final)
final_train_data, final_test_data = features.align(test_data_final_encoded, join='left', axis=1)
final_test_data.fillna(0, inplace=True)

# Predictions on test data
predictions_test = np.exp(model.predict(final_test_data))  # Inverse log transformation

# save the predictions
print("Predictions for the first 5 houses in the test data:")
print(predictions_test[:5])

predictions_file_path = os.path.join(script_dir, "predictions_test.csv")
pd.DataFrame(predictions_test).to_csv(predictions_file_path, index=False)

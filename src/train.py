import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os

# Define the path
script_dir = os.path.dirname(__file__)

# Load preprocessed data
preprocessed_data = pd.read_csv(os.path.join(script_dir, "preprocessed_train.csv"))

# Splitting data into training and validation sets
features = preprocessed_data.drop(['SalePrice'], axis=1)
target = preprocessed_data['SalePrice']
train_features, val_features, train_target, val_target = train_test_split(features, target, test_size=0.2, random_state=1)

# Model training with RandomForestRegressor
rf_model = RandomForestRegressor(random_state=1)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20]
}
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(train_features, train_target)

# Evaluate the model
val_predictions = grid_search.predict(val_features)
mae_val = mean_absolute_error(val_target, val_predictions)
print("Validation Mean Absolute Error:", mae_val)

# Save the trained model to the 'src/' directory
model_file_path = os.path.join(script_dir, "rf_model.joblib")
joblib.dump(grid_search.best_estimator_, model_file_path)

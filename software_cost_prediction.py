
# Software Project Cost Prediction

## Dataset Preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
file_path = "01. Cocomonasa_v1.csv"
df = pd.read_csv(file_path)

# Encode categorical variables
label_encoders = {}
for col in df.columns[:-2]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate features and target variable
X = df.drop(columns=["ACT_EFFORT"])
y = df["ACT_EFFORT"]

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


## Model Training and Evaluation

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {"Model": model_name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R² Score": r2}

# Initialize models
dt_model = DecisionTreeRegressor(random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Train models
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Predictions
y_pred_dt = dt_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)

# Evaluate models
dt_results = evaluate_model(y_test, y_pred_dt, "Decision Tree")
rf_results = evaluate_model(y_test, y_pred_rf, "Random Forest")
gb_results = evaluate_model(y_test, y_pred_gb, "Gradient Boosting")

# Display results
import pandas as pd
print(pd.DataFrame([dt_results, rf_results, gb_results]))


## Hyperparameter Tuning for Gradient Boosting

from sklearn.model_selection import GridSearchCV

param_grid = {"n_estimators": [100, 300, 500], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 7]}

gb_tuned = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5, scoring="r2", n_jobs=-1)
gb_tuned.fit(X_train, y_train)

best_params = gb_tuned.best_params_
gb_best_model = GradientBoostingRegressor(**best_params, random_state=42)
gb_best_model.fit(X_train, y_train)

y_pred_gb_best = gb_best_model.predict(X_test)
gb_best_results = evaluate_model(y_test, y_pred_gb_best, "Gradient Boosting (Optimized)")

# Display results
print(pd.DataFrame([dt_results, rf_results, gb_results, gb_best_results]))


### Best Parameters Found:
print(gb_best_results)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV



# Load the dataset
file_path = "data/Cocomonasa_v1.csv"
df = pd.read_csv(file_path)

# Display the first few rows to inspect the dataset
df.head()

'''
Linear Regression and Stochastic Gradient Descent (SGD) Regression 

'''
# Encode categorical variables
label_encoders = {}
for col in df.columns[:-2]:  # Exclude LOC and ACT_EFFORT which are numerical
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate features and target variable
X = df.drop(columns=["ACT_EFFORT"])
y = df["ACT_EFFORT"]

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Display the preprocessed dataset structure
print(X_train[:5], y_train[:5])


'''

Monte Carlo

'''

# Initialize models
linear_model = LinearRegression()
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)

# Train models
linear_model.fit(X_train, y_train)
sgd_model.fit(X_train, y_train)

# Predictions
y_pred_linear = linear_model.predict(X_test)
y_pred_sgd = sgd_model.predict(X_test)


# Evaluation function
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)

    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R² Score": r2
    }


# Evaluate models
linear_results = evaluate_model(y_test, y_pred_linear, "Linear Regression")
sgd_results = evaluate_model(y_test, y_pred_sgd, "SGD Regression")

# Display results
results_df = pd.DataFrame([linear_results, sgd_results])


# Visualize Monte Carlo Results

# Number of simulations
num_simulations = 10000

# Generate random samples using normal distribution (based on dataset mean & std dev)
effort_mean = y_train.mean()
effort_std = y_train.std()
simulated_efforts = np.random.normal(effort_mean, effort_std, num_simulations)

# Plot distribution of simulated efforts
plt.figure(figsize=(10, 5))
plt.hist(simulated_efforts, bins=50, alpha=0.75, edgecolor='black', density=True)
plt.axvline(effort_mean, color='red', linestyle='dashed', linewidth=2, label=f"Mean: {effort_mean:.2f}")
plt.title("Monte Carlo Simulation: Distribution of Estimated Software Effort")
plt.xlabel("Effort (man-hours)")
plt.ylabel("Probability Density")
plt.legend()
plt.show()


'''
Decision Tree Implementation (better for categorical)

'''

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

# Combine results
advanced_results_df = pd.DataFrame([dt_results, rf_results, gb_results])


# Visualization: Bar Chart for Model Errors
models = ["Decision Tree", "Random Forest", "Gradient Boosting"]
mae_values = [dt_results["MAE"], rf_results["MAE"], gb_results["MAE"]]
mse_values = [dt_results["MSE"], rf_results["MSE"], gb_results["MSE"]]
rmse_values = [dt_results["RMSE"], rf_results["RMSE"], gb_results["RMSE"]]
r2_values = [dt_results["R² Score"], rf_results["R² Score"], gb_results["R² Score"]]

# Bar plot for MAE
plt.figure(figsize=(10, 5))
plt.bar(models, mae_values, alpha=0.7, edgecolor="black")
plt.title("Mean Absolute Error (MAE) Comparison")
plt.ylabel("MAE (Lower is Better)")
plt.show()

# Bar plot for RMSE
plt.figure(figsize=(10, 5))
plt.bar(models, rmse_values, alpha=0.7, edgecolor="black", color="orange")
plt.title("Root Mean Squared Error (RMSE) Comparison")
plt.ylabel("RMSE (Lower is Better)")
plt.show()

# Bar plot for R² Score
plt.figure(figsize=(10, 5))
plt.bar(models, r2_values, alpha=0.7, edgecolor="black", color="green")
plt.title("R² Score Comparison (Higher is Better)")
plt.ylabel("R² Score")
plt.show()

# Now, let's tune the Gradient Boosting Model with hyperparameter optimization.

# Hyperparameter tuning for Gradient Boosting
param_grid = {
    "n_estimators": [100, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7]
}

# Grid search
gb_tuned = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5, scoring="r2", n_jobs=-1)
gb_tuned.fit(X_train, y_train)

# Best parameters
best_params = gb_tuned.best_params_

# Train the best model
gb_best_model = GradientBoostingRegressor(**best_params, random_state=42)
gb_best_model.fit(X_train, y_train)

# Predictions with the optimized model
y_pred_gb_best = gb_best_model.predict(X_test)

# Evaluate the optimized model
gb_best_results = evaluate_model(y_test, y_pred_gb_best, "Gradient Boosting (Optimized)")

# Append optimized results
advanced_results_df = pd.DataFrame([dt_results, rf_results, gb_results, gb_best_results])


# Return best hyperparameters found
print(best_params)
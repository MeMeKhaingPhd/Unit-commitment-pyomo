import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


#DATA CONFIGURATION # centralizes all external data inputs.

print("--- Part 0: Configuring Data Sources ---")

cleaned_solar_data_file = 'https://raw.githubusercontent.com/MeMeKhaingPhd/Unit-commitment-pyomo/refs/heads/main/solar_data_cleaned.csv' #pre-processed
url_demand_data = 'https://raw.githubusercontent.com/MeMeKhaingPhd/Unit-commitment-pyomo/refs/heads/main/dem-data-berlin.csv' #The electricity demand for the Berlin region
url_oil_east = 'https://raw.githubusercontent.com/MeMeKhaingPhd/Unit-commitment-pyomo/refs/heads/main/oil-data-east-berlin.csv' # three are the capacity data for the three conventional oil-fired power plants.
url_oil_west = 'https://raw.githubusercontent.com/MeMeKhaingPhd/Unit-commitment-pyomo/refs/heads/main/oil-data-west-berlin.csv'
url_oil_central ='https://raw.githubusercontent.com/MeMeKhaingPhd/Unit-commitment-pyomo/refs/heads/main/oil-data-central-berlin.csv'



# XGBOOST FORECASTING MODEL

print("\n--- Part 1: Training the Solar Forecasting Model ---")

try:
    df = pd.read_csv(cleaned_solar_data_file)
    print("Solar data loaded successfully from URL.")
except Exception as e:
    print(f"\nFATAL ERROR: Could not load solar data from the URL. Error: {e}")
    exit() # loads the pre-cleaned solar data from the specified URL.

#  Preprocessing 

df.rename(columns={'X50Hertz..MW.': 'Solar_MW'}, inplace=True) # target name changed because to understand well

# This part uses 'Year', 'Month', 'Day', 'Hour', 'Minute' which are correct.
df['Timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']]) # made a single time to understand model
df = df.set_index('Timestamp').sort_index()

# Create engineered time-based features # this is new features from timestamp like to give the information about day and seasonal patterns for model
df['hour'] = df.index.hour
df['dayofyear'] = df.index.dayofyear
df['month'] = df.index.month
df['year'] = df.index.year
df['dayofweek'] = df.index.dayofweek
df.ffill(inplace=True)
df.bfill(inplace=True)

target = 'Solar_MW' # output we want to predict y
features = [ # input data x
    'Temperature', 'Clearsky.DHI', 'Clearsky.DNI', 'Clearsky.GHI', 'Cloud.Type', 
    'Dew.Point', 'DHI', 'DNI', 'Fill.Flag', 'GHI', 'Ozone', 'Relative.Humidity', 
    'Solar.Zenith.Angle', 'Surface.Albedo', 'Pressure', 'Precipitable.Water', 
    'Wind.Direction', 'Wind.Speed', 'hour', 'dayofyear', 'month', 'year', 'dayofweek'
]
X = df[[f for f in features if f in df.columns]]
y = df[target]

print(f"Using {len(X.columns)} features for XGBoost model.")
split_index = int(len(df) * 0.8)
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, max_depth=5, 
                           subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
                           early_stopping_rounds=50)
xgb_reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
rmse = np.sqrt(mean_squared_error(y_test, xgb_reg.predict(X_test)))
print(f"Base XGBoost Model Trained. Test RMSE on real data: {rmse:.2f} MW")

T_horizon = 24
true_solar_generation = y_test.iloc[:T_horizon].values
print(f"Using a {T_horizon}-hour slice of TRUE solar generation for experiments.\n")


# =============================================================================
# PART 2: ECONOMIC DISPATCH MODEL DEFINITION
# =============================================================================
print("--- Part 2: Defining the Single-Node Economic Dispatch Model ---")
try:
    oil_east_data = pd.read_csv(url_oil_east).iloc[0]
    oil_west_data = pd.read_csv(url_oil_west).iloc[0]
    oil_central_data = pd.read_csv(url_oil_central).iloc[0]
    demand_df = pd.read_csv(url_demand_data)
    print("All real data files loaded successfully.")
except Exception as e:
    print(f"\nFATAL ERROR: Could not load a data file from its URL. Error: {e}")
    exit()

generators_data = [
    {"id": "Oil_East", "Pmax": oil_east_data['Capacity (MW)'], "Cost_Gen_Linear": 20, "Cost_Gen_Quadratic": 0.02},
    {"id": "Oil_West", "Pmax": oil_west_data['Capacity (MW)'], "Cost_Gen_Linear": 18, "Cost_Gen_Quadratic": 0.015},
    {"id": "Oil_Central", "Pmax": oil_central_data['Capacity (MW)'], "Cost_Gen_Linear": 25, "Cost_Gen_Quadratic": 0.03},
]
total_demand_profile = demand_df['Demand (MW)'].iloc[:T_horizon].tolist()

def create_economic_dispatch_model(hourly_demand_profile, solar_forecast):
    model = pyo.ConcreteModel(name="EconomicDispatch")
    model.I_set = pyo.Set(initialize=[gen['id'] for gen in generators_data])
    model.T_set = pyo.RangeSet(1, T_horizon)
    model.Pmax = pyo.Param(model.I_set, initialize={gen['id']: gen['Pmax'] for gen in generators_data})
    model.Cost_Linear = pyo.Param(model.I_set, initialize={gen['id']: gen['Cost_Gen_Linear'] for gen in generators_data})
    model.Cost_Quad = pyo.Param(model.I_set, initialize={gen['id']: gen['Cost_Gen_Quadratic'] for gen in generators_data})
    model.SolarPotential = pyo.Param(model.T_set, initialize={t: solar_forecast[t-1] for t in model.T_set})
    model.Demand = pyo.Param(model.T_set, initialize={t: hourly_demand_profile[t-1] for t in model.T_set})
    model.g = pyo.Var(model.I_set, model.T_set, domain=pyo.NonNegativeReals)
    model.p_curtail = pyo.Var(model.T_set, domain=pyo.NonNegativeReals)
    def total_cost_rule(m):
        return sum(m.Cost_Linear[i] * m.g[i, t] + m.Cost_Quad[i] * (m.g[i, t]**2) for i in m.I_set for t in m.T_set)
    model.TotalCost = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)
    def power_balance_rule(m, t):
        return sum(m.g[i, t] for i in m.I_set) + (m.SolarPotential[t] - m.p_curtail[t]) == m.Demand[t]
    model.PowerBalance = pyo.Constraint(model.T_set, rule=power_balance_rule)
    model.GenLimits = pyo.Constraint(model.I_set, model.T_set, rule=lambda m, i, t: m.g[i, t] <= m.Pmax[i])
    model.CurtailmentLimits = pyo.Constraint(model.T_set, rule=lambda m, t: m.p_curtail[t] <= m.SolarPotential[t])
    return model

try:
    solver = pyo.SolverFactory('gurobi')
except Exception:
    print("Gurobi not found, falling back to CBC.")
    solver = pyo.SolverFactory('cbc')


# =============================================================================
# PART 3: RUNNING SIMULATIONS
# =============================================================================
print("\n--- Part 3: Running Simulation Experiments ---")

### FIXED ###
# This is the complete and correct runner function.
def run_dispatch_and_get_results(solar_forecast):
    model = create_economic_dispatch_model(total_demand_profile, solar_forecast)
    results = solver.solve(model, tee=False)
    # This check is crucial. It prevents the script from crashing on failure.
    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        return {'status': str(results.solver.termination_condition), 'cost': np.nan}
    # This line is only reached if the solver succeeds.
    return {'status': 'optimal', 'cost': pyo.value(model.TotalCost)}

# --- 3A: Economic Insights (Post-Processing) ---
print("\n--- Experiment 3A: Calculating Economic Loss from Accuracy Variations ---")
perfect_foresight_results = run_dispatch_and_get_results(true_solar_generation)
if perfect_foresight_results.get('status') != 'optimal':
    print(f"FATAL ERROR: Could not solve the perfect foresight case. Status: {perfect_foresight_results.get('status')}")
    exit()
PERFECT_COST = perfect_foresight_results['cost']
print(f"Calculated 'Perfect Foresight' Benchmark Cost: ${PERFECT_COST:,.0f}")

results_rmse = []
target_rmses = np.linspace(50, 500, 15)
for r in target_rmses:
    noise = np.random.normal(loc=0, scale=r, size=T_horizon)
    noisy_forecast = np.maximum(0, true_solar_generation + noise)
    res = run_dispatch_and_get_results(noisy_forecast)
    if res.get('status') == 'optimal':
        res['rmse'] = np.sqrt(mean_squared_error(true_solar_generation, noisy_forecast))
        res['economic_loss'] = res['cost'] - PERFECT_COST
        results_rmse.append(res)
df_rmse = pd.DataFrame(results_rmse)
print("Finished RMSE vs. Economic Loss experiment.")

# --- 3B: Asymmetric Accuracy Objectives (Custom Losses) ---
print("\n--- Experiment 3B: Training with Custom Asymmetric Losses ---")

def asymmetric_squared_error(y_true, y_pred, weight_under=5.0, weight_over=1.0):
    residual = y_pred - y_true
    grad = np.where(residual < 0, 2 * weight_under * residual, 2 * weight_over * residual)
    hess = np.where(residual < 0, 2 * weight_under, 2 * weight_over)
    return grad, hess

# We can also create a version that penalizes over-forecasting
def penalize_over_forecasting(y_true, y_pred):
    return asymmetric_squared_error(y_true, y_pred, weight_under=1.0, weight_over=5.0)

asymmetric_models_to_test = {
    "Penalize Under-Forecast (W=5)": asymmetric_squared_error,
    "Penalize Over-Forecast (W=5)": penalize_over_forecasting
}
results_asymmetric = []
for name, custom_objective in asymmetric_models_to_test.items():
    print(f"Training XGBoost model with Custom Loss: {name}...")
    model = xgb.XGBRegressor(objective=custom_objective, n_estimators=1000,
                               learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1,
                               early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    predictions = model.predict(X_test)
    dispatch_forecast = np.maximum(0, predictions[:T_horizon])
    dispatch_results = run_dispatch_and_get_results(dispatch_forecast)
    if dispatch_results.get('status') == 'optimal':
        results_asymmetric.append({
            'Model Type': name,
            'Cost': dispatch_results['cost'],
            'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
            'MBE': np.mean(predictions - y_test) # Mean Bias Error
        })
df_asymmetric = pd.DataFrame(results_asymmetric)
print("Finished custom loss training experiment.")

# --- 3C: Standard Loss Comparison (Built-in losses) ---
print("\n--- Experiment 3C: Comparing Standard ML Losses (MSE, MAE, etc.) ---")
standard_losses_to_test = {
    'Standard MSE (L2)': 'reg:squarederror',
    'Standard MAE (L1)': 'reg:absoluteerror',
    'Standard Huber': 'reg:pseudohubererror'
}
results_standard_loss = []
for name, objective_func in standard_losses_to_test.items():
    print(f"Training XGBoost model with Standard Loss: {name}...")
    model = xgb.XGBRegressor(objective=objective_func, n_estimators=1000,
                               learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1,
                               early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    predictions = model.predict(X_test)
    dispatch_forecast = np.maximum(0, predictions[:T_horizon])
    dispatch_results = run_dispatch_and_get_results(dispatch_forecast)
    if dispatch_results.get('status') == 'optimal':
        results_standard_loss.append({
            'Model Type': name,
            'Cost': dispatch_results['cost'],
            'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
            'MBE': np.mean(predictions - y_test)
        })
df_standard_loss = pd.DataFrame(results_standard_loss)
print("Finished standard loss comparison experiment.")

# Combine results from 3B and 3C for plotting
df_all_models = pd.concat([df_asymmetric, df_standard_loss], ignore_index=True)


# =============================================================================
# PART 4: VISUALIZING ALL RESULTS
# =============================================================================
print("\n--- Part 4: Visualizing All Results ---")
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(22, 16))
fig.suptitle('Comprehensive Analysis of Forecast Accuracy and Objectives on System Economics', fontsize=20)

# --- Plot 1: Economic Loss vs. RMSE (from 3A) ---
ax1 = axes[0, 0]
if not df_rmse.empty:
    sns.regplot(x='rmse', y='economic_loss', data=df_rmse, ax=ax1, color='orangered',
                scatter_kws={'alpha': 0.6, 's': 50}, line_kws={'linestyle':'--'})
    ax1.set_title('Economic Loss from Forecast Inaccuracy', fontsize=16)
    ax1.set_xlabel('Forecast RMSE (MW)', fontsize=12)
    ax1.set_ylabel('Economic Loss ($)', fontsize=12)
    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'${format(int(x), ",")}'))
    ax1.axhline(0, color='black', lw=1, linestyle='-')

# --- Plot 2: Scatter plot of all trained models (from 3B & 3C) ---
ax2 = axes[0, 1]
if not df_all_models.empty:
    sns.scatterplot(x='RMSE', y='Cost', data=df_all_models, hue='Model Type', style='Model Type',
                    s=200, ax=ax2, palette='viridis')
    ax2.set_title('Performance of Models Trained with Different Objectives', fontsize=16)
    ax2.set_xlabel('Resulting Forecast RMSE (MW)', fontsize=12)
    ax2.set_ylabel('Resulting System Cost ($)', fontsize=12)
    ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'${format(int(x), ",")}'))
    ax2.legend(title='Training Objective', bbox_to_anchor=(1.05, 1), loc='upper left')

# --- Plot 3: Bar chart comparing Costs (from 3B & 3C) ---
ax3 = axes[1, 0]
if not df_all_models.empty:
    df_all_models_sorted = df_all_models.sort_values('Cost')
    sns.barplot(x='Cost', y='Model Type', data=df_all_models_sorted, ax=ax3, palette='plasma')
    ax3.set_title('Economic Performance Ranking of Models', fontsize=16)
    ax3.set_xlabel('Total System Cost ($)', fontsize=12)
    ax3.set_ylabel('Training Objective', fontsize=12)
    ax3.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'${format(int(x), ",")}'))

# --- Plot 4: Bar chart comparing Bias (from 3B & 3C) ---
ax4 = axes[1, 1]
if not df_all_models.empty:
    sns.barplot(x='MBE', y='Model Type', data=df_all_models.sort_values('Cost'), ax=ax4, palette='plasma')
    ax4.set_title('Resulting Forecast Bias (MBE) of Models', fontsize=16)
    ax4.set_xlabel('Mean Bias Error (MW)', fontsize=12)
    ax4.set_ylabel('') # No label to avoid clutter
    ax4.axvline(0, color='black', lw=1, linestyle='--')

plt.tight_layout(rect=[0, 0, 0.9, 0.96])
plt.show()

print("\nScript finished successfully.")
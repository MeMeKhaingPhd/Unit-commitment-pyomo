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

print("--- Part 0: Configuring Data Sources ---")
cleaned_solar_data_file = 'https://raw.githubusercontent.com/MeMeKhaingPhd/Unit-commitment-pyomo/main/solar_data_cleaned.csv'
url_demand_data = 'https://raw.githubusercontent.com/MeMeKhaingPhd/Unit-commitment-pyomo/main/dem-data-berlin.csv'
url_oil_east = 'https://raw.githubusercontent.com/MeMeKhaingPhd/Unit-commitment-pyomo/main/oil-data-east-berlin.csv'
url_oil_west = 'https://raw.githubusercontent.com/MeMeKhaingPhd/Unit-commitment-pyomo/main/oil-data-west-berlin.csv'
url_oil_central ='https://raw.githubusercontent.com/MeMeKhaingPhd/Unit-commitment-pyomo/main/oil-data-central-berlin.csv'


# XGBOOST FORECASTING MODEL

print("\n--- Part 1: Training the Solar Forecasting Model ---")

try:
    df = pd.read_csv(cleaned_solar_data_file)
    print("Solar data loaded successfully from URL.")
except Exception as e:
    print(f"\nFATAL ERROR: Could not load solar data from the URL. Error: {e}")
    exit() # loads the pre-cleaned solar data from the specified URL.

#  Preprocessing 

df.rename(columns={'X50Hertz..MW.': 'Solar_MW'}, inplace=True) 


df['Timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']]) 
df = df.set_index('Timestamp').sort_index()


df['hour'] = df.index.hour
df['dayofyear'] = df.index.dayofyear
df['month'] = df.index.month
df['year'] = df.index.year
df['dayofweek'] = df.index.dayofweek
df.ffill(inplace=True)
df.bfill(inplace=True)
target = 'Solar_MW'
features = [
    'Temperature', 'Clearsky.DHI', 'Clearsky.DNI', 'Clearsky.GHI', 'Cloud.Type', 
    'Dew.Point', 'DHI', 'DNI', 'Fill.Flag', 'GHI', 'Ozone', 'Relative.Humidity', 
    'Solar.Zenith.Angle', 'Surface.Albedo', 'Pressure', 'Precipitable.Water', 
    'Wind.Direction', 'Wind.Speed', 'hour', 'dayofyear', 'month', 'year', 'dayofweek'
]
X = df[[f for f in features if f in df.columns]]
y = df[target]
split_index = int(len(df) * 0.8)
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
T_horizon = 10
true_solar_generation = y_test.iloc[:T_horizon].values
print(f"Data prepared. Using a {T_horizon}-hour slice for experiments.\n")


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
    {"id": "Oil_East", "bus": "Bus1", "Pmin": 0, "Pmax": oil_east_data['Capacity (MW)'], "Cost_Gen_Linear": 20, "Cost_Gen_Quadratic": 0.02},
    {"id": "Oil_West", "bus": "Bus2", "Pmin": 0, "Pmax": oil_west_data['Capacity (MW)'], "Cost_Gen_Linear": 18, "Cost_Gen_Quadratic": 0.015},
    {"id": "Oil_Central", "bus": "Bus3", "Pmin": 0, "Pmax": oil_central_data['Capacity (MW)'], "Cost_Gen_Linear": 25, "Cost_Gen_Quadratic": 0.03},
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

def run_dispatch_and_get_results(solar_forecast):
    model = create_economic_dispatch_model(total_demand_profile, solar_forecast)
    results = solver.solve(model, tee=False)
    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        return {'status': str(results.solver.termination_condition), 'cost': np.nan}
    return {'status': 'optimal', 'cost': pyo.value(model.TotalCost)}



#  EXPERIMENTS ON FORECAST MODEL PARAMETERS

print("\n--- Part 3: Running New Experiments on Forecast Model ---")

### --- Experiment A: Varying Number of Training Rounds (`n_estimators`) ---
print("\n--- Experiment A: Impact of Training Rounds (n_estimators) ---")
results_estimators = []
estimator_steps = range(10, 201, 10)

for n_est in estimator_steps:
    print(f"Training XGBoost model with n_estimators = {n_est}...")
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=n_est, 
                               learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train, verbose=False)
    predictions = model.predict(X_test)
    forecast_rmse = np.sqrt(mean_squared_error(y_test, predictions))
    dispatch_forecast = predictions[:T_horizon]
    dispatch_results = run_dispatch_and_get_results(dispatch_forecast)
    if dispatch_results.get('status') == 'optimal':
        results_estimators.append({
            'n_estimators': n_est,
            'forecast_rmse': forecast_rmse,
            'system_cost': dispatch_results['cost']
        })
        print(f"  -> Forecast RMSE: {forecast_rmse:7.2f} | System Cost: ${dispatch_results['cost']:,.0f}")
df_estimators = pd.DataFrame(results_estimators)


# Comparing Different Loss Functions 
print("\n--- Experiment B: Impact of Different Loss Functions ---")
results_loss_functions = []
loss_functions_to_test = {
    'Squared Error (L2)': 'reg:squarederror',
    'Huber Loss': 'reg:pseudohubererror',
    'Absolute Error (L1)': 'reg:absoluteerror'
}
for name, objective_func in loss_functions_to_test.items():
    print(f"Training XGBoost model with Loss Function: {name}...")
    model = xgb.XGBRegressor(objective=objective_func, n_estimators=1000, 
                               learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1,
                               early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    predictions = model.predict(X_test)
    forecast_rmse = np.sqrt(mean_squared_error(y_test, predictions))
    dispatch_forecast = predictions[:T_horizon]
    dispatch_results = run_dispatch_and_get_results(dispatch_forecast)
    if dispatch_results.get('status') == 'optimal':
        results_loss_functions.append({
            'loss_function': name,
            'n_estimators_used': model.best_iteration,
            'forecast_rmse': forecast_rmse,
            'system_cost': dispatch_results['cost']
        })
        print(f"  -> Model used {model.best_iteration} estimators.")
        print(f"  -> Forecast RMSE: {forecast_rmse:7.2f} | System Cost: ${dispatch_results['cost']:,.0f}")
df_loss = pd.DataFrame(results_loss_functions)

print("\n--- Part 4: Visualizing New Experiment Results ---")
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Analysis of Forecast Model Hyperparameters on System Performance', fontsize=18)

# --- Plots for Experiment A: n_estimators ---
ax1 = axes[0, 0]
if not df_estimators.empty:
    sns.lineplot(x='n_estimators', y='forecast_rmse', data=df_estimators, ax=ax1, marker='o', color='blue')
    ax1.set_title('Impact of Training Rounds on Forecast Accuracy', fontsize=14)
    ax1.set_xlabel('Number of Estimators (Training Rounds)')
    ax1.set_ylabel('Forecast RMSE (MW)')
    ax1.grid(True, linestyle='--')

ax2 = axes[0, 1]
if not df_estimators.empty:
    sns.lineplot(x='n_estimators', y='system_cost', data=df_estimators, ax=ax2, marker='o', color='green')
    ax2.set_title('Impact of Training Rounds on Total System Cost', fontsize=14)
    ax2.set_xlabel('Number of Estimators (Training Rounds)')
    ax2.set_ylabel('Total System Cost ($)')
    ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'${format(int(x), ",")}'))
    ax2.grid(True, linestyle='--')

# Loss Functions 
ax3 = axes[1, 0]
if not df_loss.empty:
    df_loss_sorted = df_loss.sort_values('system_cost').reset_index()
    ax3_twin = ax3.twinx()
    sns.barplot(x='loss_function', y='system_cost', data=df_loss_sorted, ax=ax3, palette="viridis", alpha=0.7)
    ax3.set_title('Comparison of Different Loss Functions', fontsize=14)
    ax3.set_xlabel('XGBoost Loss Function', fontsize=12)
    ax3.set_ylabel('Total System Cost ($)', fontsize=12, color='darkgreen')
    ax3.tick_params(axis='y', labelcolor='darkgreen')
    ax3.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'${format(int(x), ",")}'))
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=15, ha="right")
    sns.lineplot(x='loss_function', y='forecast_rmse', data=df_loss_sorted, ax=ax3_twin,
                 marker='o', color='crimson', sort=False, label='Forecast RMSE')
    ax3_twin.set_ylabel('Forecast RMSE (MW)', fontsize=12, color='crimson')
    ax3_twin.tick_params(axis='y', labelcolor='crimson')
    ax3_twin.grid(False)

# Forecast Accuracy and System Cost 
ax4 = axes[1, 1]
if not df_estimators.empty:
    sns.scatterplot(x='forecast_rmse', y='system_cost', data=df_estimators, ax=ax4,
                    hue='n_estimators', palette='coolwarm', s=100, alpha=0.8)
    ax4.set_title('Forecast Accuracy vs. System Cost', fontsize=14)
    ax4.set_xlabel('Forecast RMSE (MW)')
    ax4.set_ylabel('Total System Cost ($)')
    ax4.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'${format(int(x), ",")}'))
    ax4.grid(True, linestyle='--')
    handles, labels = ax4.get_legend_handles_labels()
    ax4.legend(handles, labels, title='n_estimators', bbox_to_anchor=(1.05, 1), loc='upper left')


plt.subplots_adjust(hspace=0.5, wspace=0.3)

plt.show()

print("\nScript finished successfully.")
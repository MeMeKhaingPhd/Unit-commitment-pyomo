import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.callback import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


#DATA CONFIGURATION

print("--- Part 0: Configuring Data Sources ---")

cleaned_solar_data_file = 'https://raw.githubusercontent.com/MeMeKhaingPhd/Unit-commitment-pyomo/refs/heads/main/solar_data_cleaned.csv'
url_demand_data = 'https://raw.githubusercontent.com/MeMeKhaingPhd/Unit-commitment-pyomo/refs/heads/main/dem-data-berlin.csv'
url_oil_east = 'https://raw.githubusercontent.com/MeMeKhaingPhd/Unit-commitment-pyomo/refs/heads/main/oil-data-east-berlin.csv'
url_oil_west = 'https://raw.githubusercontent.com/MeMeKhaingPhd/Unit-commitment-pyomo/refs/heads/main/oil-data-west-berlin.csv'
url_oil_central ='https://raw.githubusercontent.com/MeMeKhaingPhd/Unit-commitment-pyomo/refs/heads/main/oil-data-central-berlin.csv'



# XGBOOST FORECASTING MODEL

print("\n--- Part 1: Training the Solar Forecasting Model ---")

try:
    df = pd.read_csv(cleaned_solar_data_file)
    print("Solar data loaded successfully from URL.")
except Exception as e:
    print(f"\nFATAL ERROR: Could not load solar data from the URL. Error: {e}")
    exit()

#  Preprocessing 

df.rename(columns={'X50Hertz..MW.': 'Solar_MW'}, inplace=True)

# This part uses 'Year', 'Month', 'Day', 'Hour', 'Minute' which are correct.
df['Timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
df = df.set_index('Timestamp').sort_index()

# Create engineered time-based features
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
X = df[[f for f in features if f in df.columns]].copy()
y = df[target].copy()

X.ffill(inplace=True)
X.bfill(inplace=True)

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

T_horizon = 10
true_solar_generation = y_test.iloc[:T_horizon].values
print(f"Using a {T_horizon}-hour slice of TRUE solar generation for experiments.\n")



#  ECONOMIC DISPATCH (ED) MODEL DEFINITION

print("--- Part 2: Defining the Single-Node Power System and ED Model ---")

try:
    oil_east_data = pd.read_csv(url_oil_east).iloc[0]
    oil_west_data = pd.read_csv(url_oil_west).iloc[0]
    oil_central_data = pd.read_csv(url_oil_central).iloc[0]
    demand_df = pd.read_csv(url_demand_data)
    print("All real data files loaded successfully.")
except Exception as e:
    print(f"\nFATAL ERROR: Could not load a data file from its URL. Error: {e}")
    exit()

capacity_scaling_factor = 10
generators_data = [
    {"id": "Oil_East", "Pmin": 0, "Pmax": oil_east_data['Capacity (MW)'] * capacity_scaling_factor, "Cost_Gen_Linear": 20, "Cost_Gen_Quadratic": 0.02, "Emission_Rate": 0.25},
    {"id": "Oil_West", "Pmin": 0, "Pmax": oil_west_data['Capacity (MW)'] * capacity_scaling_factor, "Cost_Gen_Linear": 18, "Cost_Gen_Quadratic": 0.015, "Emission_Rate": 0.20},
    {"id": "Oil_Central", "Pmin": 0, "Pmax": oil_central_data['Capacity (MW)'] * capacity_scaling_factor, "Cost_Gen_Linear": 25, "Cost_Gen_Quadratic": 0.03, "Emission_Rate": 0.30},
]
print(f"NOTE: Conventional generator capacities scaled by {capacity_scaling_factor} to be feasible for real demand.")
total_demand_profile = demand_df['Demand (MW)'].iloc[:T_horizon].tolist()

def create_economic_dispatch_model(demand_profile, solar_forecast):
    model = pyo.ConcreteModel(name="EconomicDispatch")
    model.I_conv = pyo.Set(initialize=[gen['id'] for gen in generators_data])
    model.T = pyo.RangeSet(1, T_horizon)
    model.Pmax = pyo.Param(model.I_conv, initialize={gen['id']: gen['Pmax'] for gen in generators_data})
    model.Cost_Linear = pyo.Param(model.I_conv, initialize={gen['id']: gen['Cost_Gen_Linear'] for gen in generators_data})
    model.Cost_Quad = pyo.Param(model.I_conv, initialize={gen['id']: gen['Cost_Gen_Quadratic'] for gen in generators_data})
    model.Demand = pyo.Param(model.T, initialize={t: demand_profile[t-1] for t in model.T})
    model.Solar_Available = pyo.Param(model.T, initialize={t: solar_forecast[t-1] for t in model.T})
    model.g_conv = pyo.Var(model.I_conv, model.T, domain=pyo.NonNegativeReals)
    model.p_curtail = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    def total_cost_rule(m):
        return sum(m.Cost_Linear[i] * m.g_conv[i,t] + m.Cost_Quad[i] * (m.g_conv[i,t]**2) 
                   for i in m.I_conv for t in m.T)
    model.TotalCost = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)
    def power_balance_rule(m, t):
        total_conventional_gen = sum(m.g_conv[i, t] for i in m.I_conv)
        solar_gen_used = m.Solar_Available[t] - m.p_curtail[t]
        return total_conventional_gen + solar_gen_used == m.Demand[t]
    model.PowerBalance = pyo.Constraint(model.T, rule=power_balance_rule)
    def gen_limits_rule(m, i, t):
        return m.g_conv[i,t] <= m.Pmax[i]
    model.GenLimits = pyo.Constraint(model.I_conv, model.T, rule=gen_limits_rule)
    def curtailment_limits_rule(m, t):
        return m.p_curtail[t] <= m.Solar_Available[t]
    model.CurtailmentLimits = pyo.Constraint(model.T, rule=curtailment_limits_rule)
    return model

try:
    solver = pyo.SolverFactory('gurobi')
except Exception:
    print("Gurobi not found, falling back to CBC. This may be slow.")
    solver = pyo.SolverFactory('cbc')


# RUNNING THE TWO-LEVEL SIMULATION

print("\n--- Part 3: Running Two-Level Simulation Experiments ---")

def run_dispatch_and_get_results(solar_forecast):
    model = create_economic_dispatch_model(total_demand_profile, solar_forecast)
    results = solver.solve(model, tee=False)
    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        return {'status': str(results.solver.termination_condition)}
    dispatch = {gen['id']: [pyo.value(model.g_conv[gen['id'], t]) for t in model.T] for gen in generators_data}
    dispatch['Solar_Used'] = [pyo.value(model.Solar_Available[t] - model.p_curtail[t]) for t in model.T]
    dispatch['Curtailment'] = [pyo.value(model.p_curtail[t]) for t in model.T]
    return {'status': 'optimal', 'cost': pyo.value(model.TotalCost), 'dispatch_df': pd.DataFrame(dispatch)}

print("\n--- Running Base Case Analysis (Median Forecast) ---")
base_case_results = run_dispatch_and_get_results(true_solar_generation)

print("\n--- Running Experiment: Impact of Forecast RMSE on Cost ---")
results_rmse = []
target_rmses = np.linspace(50, 500, 10)
for r in target_rmses:
    noise = np.random.normal(loc=0, scale=r, size=T_horizon)
    noisy_forecast = np.maximum(0, true_solar_generation + noise)
    res = run_dispatch_and_get_results(noisy_forecast)
    if res['status'] == 'optimal':
        res['rmse'] = np.sqrt(mean_squared_error(true_solar_generation, noisy_forecast))
        results_rmse.append(res)
        print(f"Target RMSE: {r:6.1f} | Actual RMSE: {res['rmse']:6.1f} | Cost: ${res.get('cost', 0):,.0f}")
df_rmse = pd.DataFrame(results_rmse)

#  Experiment Impact of Forecast Bias 
print("\n--- Running Experiment: Impact of Forecast Bias on Cost ---")
results_bias = []
mean_solar = true_solar_generation.mean()
# Use a reasonable range of bias, +/- 50% of the average solar generation
bias_levels = np.linspace(-0.5 * mean_solar, 0.5 * mean_solar, 11)
for bias in bias_levels:
    biased_forecast = np.maximum(0, true_solar_generation + bias)
    res = run_dispatch_and_get_results(biased_forecast)
    if res['status'] == 'optimal':
        res['bias'] = bias
        results_bias.append(res)
        print(f"Bias: {bias:6.1f} MW | Status: {res['status']:<12} | Cost: ${res.get('cost', 0):,.0f}")
df_bias = pd.DataFrame(results_bias)



# VISUALIZATION

print("\n--- Part 4: Visualizing Results ---")
sns.set_style("whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Two-Level Economic Dispatch Simulation Results', fontsize=18)

#Dispatch Stack for the Base Case 
ax1 = axes[0, 0]
if base_case_results.get('status') == 'optimal':
    dispatch_df = base_case_results['dispatch_df']
    gens = [g['id'] for g in generators_data]
    colors = ['#FFC300', '#FF5733', '#C70039']
    bottom = np.zeros(T_horizon)
    for i, gen_id in enumerate(gens):
        ax1.bar(range(T_horizon), dispatch_df[gen_id], bottom=bottom, label=gen_id, color=colors[i])
        bottom += dispatch_df[gen_id]
    ax1.bar(range(T_horizon), dispatch_df['Solar_Used'], bottom=bottom, label='Solar Used', color='#2ECC71')
    ax1.plot(range(T_horizon), total_demand_profile, 'k--', label='Total Demand', linewidth=2)
    ax1.set_title('Base Case: Hourly Generation Dispatch Stack', fontsize=14)
    ax1.set_xlabel('Hour of the Day')
    ax1.set_ylabel('Power (MW)')
    ax1.set_xticks(range(0, T_horizon, 2))
    ax1.grid(True, axis='y', linestyle='--')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(range(T_horizon), dispatch_df['Curtailment'], color='red', linestyle=':', marker='o', markersize=4, label='Curtailment')
    ax1_twin.set_ylabel('Curtailment (MW)', color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    ax1_twin.set_ylim(bottom=0)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

# Cost vs. Forecast Error (RMSE) 
ax2 = axes[0, 1]
if not df_rmse.empty:
    sns.regplot(x='rmse', y='cost', data=df_rmse, ax=ax2, line_kws={"color": "red"})
    ax2.set_title('Impact of Forecast RMSE on Total System Cost', fontsize=14)
    ax2.set_xlabel('Forecast RMSE (MW)')
    ax2.set_ylabel('Total System Cost ($)')
    ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'${format(int(x), ",")}'))
    ax2.grid(True, linestyle='--')

# Cost vs. Forecast Bias (Bottom-Left) 
ax3 = axes[1, 0]
if not df_bias.empty:
    sns.regplot(x='bias', y='cost', data=df_bias, ax=ax3, order=2, line_kws={"color": "purple"}, scatter_kws={'alpha':0.6})
    ax3.set_title('Impact of Forecast Bias on Total System Cost', fontsize=14)
    ax3.set_xlabel('Forecast Bias (MW)')
    ax3.set_ylabel('Total System Cost ($)')
    ax3.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'${format(int(x), ",")}'))
    ax3.grid(True, linestyle='--')


axes[1, 1].axis('off')


plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.show()

print("\nScript finished successfully.")
import pandas as pd
import numpy as np
import xgboost as xgb
# ADDED: Import the EarlyStopping callback for newer XGBoost versions
from xgboost.callback import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# DATA CONFIGURATION

print("--- Part 0: Configuring Data Sources ---")

cleaned_solar_data_file = 'https://raw.githubusercontent.com/MeMeKhaingPhd/Unit-commitment-pyomo/refs/heads/main/solar_data_cleaned.csv'
url_demand_data = 'https://raw.githubusercontent.com/MeMeKhaingPhd/Unit-commitment-pyomo/refs/heads/main/dem-data-berlin.csv'
url_oil_east = 'https://raw.githubusercontent.com/MeMeKhaingPhd/Unit-commitment-pyomo/refs/heads/main/oil-data-east-berlin.csv'
url_oil_west = 'https://raw.githubusercontent.com/MeMeKhaingPhd/Unit-commitment-pyomo/refs/heads/main/oil-data-west-berlin.csv'
url_oil_central ='https://raw.githubusercontent.com/MeMeKhaingPhd/Unit-commitment-pyomo/refs/heads/main/oil-data-central-berlin.csv'



# 1: XGBOOST FORECASTING MODEL

print("\n--- Part 1: Training the Solar Forecasting Model ---")

try:
    df = pd.read_csv(cleaned_solar_data_file)
    print("Solar data loaded successfully from URL.")
except Exception as e:
    print(f"\nFATAL ERROR: Could not load solar data from the URL. Error: {e}")
    exit()

# Preprocessing 

df.rename(columns={'X50Hertz..MW.': 'Solar_MW'}, inplace=True)

# This part uses 'Year', 'Month', 'Day', 'Hour', 'Minute' which are correct
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
    'Temperature', 'Clearsky.DHI', 'Clearsky.DNI', 'Clearsky.GHI',
    'Cloud.Type', 'Dew.Point', 'DHI', 'DNI', 'Fill.Flag', 'GHI', 'Ozone',
    'Relative.Humidity', 'Solar.Zenith.Angle', 'Surface.Albedo',
    'Pressure', 'Precipitable.Water', 'Wind.Direction', 'Wind.Speed',
   # include the engineered time features we just created
    'hour', 'dayofyear', 'month', 'year', 'dayofweek'
]
# This line of code is robust and will only use the features that exist in the dataframe
X = df[[f for f in features if f in df.columns]]
y = df[target]

print(f"Using {len(X.columns)} features for XGBoost model.")

split_index = int(len(df) * 0.8)
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]

xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_reg.fit(X_train, y_train)
rmse = np.sqrt(mean_squared_error(y_test, xgb_reg.predict(X_test)))
print(f"Base XGBoost Model Trained. Test RMSE on scaled data: {rmse:.2f} MW")

T_horizon = 10
# THIS IS THE FINAL FIX: Scale solar generation to a realistic level for the grid
true_solar_generation = y_test.iloc[:T_horizon].values / 10.0
print(f"NOTE: True solar generation has been scaled down by a factor of 10 to be feasible for the grid.")
print(f"Using a {T_horizon}-hour slice of true solar generation for experiments.\n")


# 2: OPTIMAL POWER FLOW (OPF) MODEL DEFINITION

print("--- Part 2: Defining the Power System and OPF Model from Data URLs ---")

BaseMVA = 100.0
nodes_data = {"Bus1": {"is_reference": True}, "Bus2": {}, "Bus3": {}}
lines_data = {
    ("Bus1", "Bus2"): {"reactance_pu": 0.1, "capacity_mw": 200},
    ("Bus1", "Bus3"): {"reactance_pu": 0.08, "capacity_mw": 150},
    ("Bus2", "Bus3"): {"reactance_pu": 0.05, "capacity_mw": 180},
}
SOLAR_BUS = "Bus2"

try:
    oil_east_data = pd.read_csv(url_oil_east).iloc[0]
    oil_west_data = pd.read_csv(url_oil_west).iloc[0]
    oil_central_data = pd.read_csv(url_oil_central).iloc[0]
    print("Oil generator data loaded successfully from URLs.")
    
    demand_df = pd.read_csv(url_demand_data)
    print("Demand data loaded successfully from URL.")
    demand_df['Demand (MW)'] = demand_df['Demand (MW)'] / 6.0
    print("NOTE: Demand has been scaled down by a factor of 5 to ensure model is feasible.")
except Exception as e:
    print(f"\nFATAL ERROR: Could not load a data file from its URL. Error: {e}")
    exit()

generators_data = [
    {
        "id": "Oil_East", "bus": "Bus1", "Pmin": 0,
        "Pmax": oil_east_data['Capacity (MW)'],
        "Cost_Gen_Linear": 20, "Cost_Gen_Quadratic": 0.02, "Emission_Rate": 0.25
    },
    {
        "id": "Oil_West", "bus": "Bus2", "Pmin": 0,
        "Pmax": oil_west_data['Capacity (MW)'],
        "Cost_Gen_Linear": 18, "Cost_Gen_Quadratic": 0.015, "Emission_Rate": 0.20
    },
    {
        "id": "Oil_Central", "bus": "Bus3", "Pmin": 0,
        "Pmax": oil_central_data['Capacity (MW)'],
        "Cost_Gen_Linear": 25, "Cost_Gen_Quadratic": 0.03, "Emission_Rate": 0.30
    },
]

try:
    total_demand_profile = demand_df['Demand (MW)'].iloc[:T_horizon].tolist()
    if len(total_demand_profile) < T_horizon:
        print(f"\nWARNING: Demand data has fewer than {T_horizon} hours. Simulation may be affected.")
except KeyError:
    print("\nFATAL ERROR: Column 'Demand (MW)' not found in the demand data file. Please check the column name.")
    exit()

nodal_demand_fractions = {"Bus1": 0.2, "Bus2": 0.5, "Bus3": 0.3}

def create_opf_model(hourly_demand, solar_forecast, emission_params):
    model = pyo.ConcreteModel(name="TwoLevel_OPF")
    model.I_set = pyo.Set(initialize=[gen['id'] for gen in generators_data])
    model.T_set = pyo.RangeSet(1, T_horizon)
    model.B_nodes = pyo.Set(initialize=nodes_data.keys())
    model.L_lines = pyo.Set(initialize=lines_data.keys(), dimen=2)
    model.Pmax = pyo.Param(model.I_set, initialize={gen['id']: gen['Pmax'] for gen in generators_data})
    model.Cost_Gen_Linear = pyo.Param(model.I_set, initialize={gen['id']: gen['Cost_Gen_Linear'] for gen in generators_data})
    model.Cost_Gen_Quadratic = pyo.Param(model.I_set, initialize={gen['id']: gen['Cost_Gen_Quadratic'] for gen in generators_data})
    model.GenBus = pyo.Param(model.I_set, initialize={gen['id']: gen['bus'] for gen in generators_data})
    model.LineReactance = pyo.Param(model.L_lines, initialize={line: data['reactance_pu'] for line, data in lines_data.items()})
    model.LineCapacity = pyo.Param(model.L_lines, initialize={line: data['capacity_mw'] for line, data in lines_data.items()})
    model.BaseMVA = pyo.Param(initialize=BaseMVA)
    model.ReferenceBus = pyo.Param(initialize=next(b for b, d in nodes_data.items() if d.get("is_reference")))
    model.SolarPotential = pyo.Param(model.T_set, initialize={t: solar_forecast[t-1] for t in model.T_set})
    model.GeneratorsAtBus = pyo.Set(model.B_nodes, initialize=lambda m, b: [i for i in m.I_set if m.GenBus[i] == b])
    model.LinesFromBus = pyo.Set(model.B_nodes, initialize=lambda m, b: [l for l in m.L_lines if l[0] == b])
    model.LinesToBus = pyo.Set(model.B_nodes, initialize=lambda m, b: [l for l in m.L_lines if l[1] == b])
   #The GenLimits constraint handle the upper bound
    model.g = pyo.Var(model.I_set, model.T_set, domain=pyo.NonNegativeReals)
    model.theta = pyo.Var(model.B_nodes, model.T_set, domain=pyo.Reals)
    model.pline = pyo.Var(model.L_lines, model.T_set, domain=pyo.Reals)
    
    def total_cost_rule(m):
        gen_cost = sum(m.Cost_Gen_Linear[i] * m.g[i,t] + m.Cost_Gen_Quadratic[i] * (m.g[i,t]**2) for i in m.I_set for t in m.T_set)
        emission_cost = emission_params['carbon_price'] * sum(emission_params['rates'][i] * m.g[i,t] for i in m.I_set for t in m.T_set)
        return gen_cost + emission_cost
    model.TotalCost = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

    def power_balance_rule(m, b, t):
        generation_at_bus = sum(m.g[i,t] for i in m.GeneratorsAtBus[b])
        demand_at_bus = hourly_demand[b][t-1]
        solar_injection = m.SolarPotential[t] if b == SOLAR_BUS else 0
        flow_out = sum(m.pline[line,t] for line in m.LinesFromBus[b])
        flow_in = sum(m.pline[line,t] for line in m.LinesToBus[b])
        return generation_at_bus + solar_injection - demand_at_bus == flow_out - flow_in
    model.PowerBalance = pyo.Constraint(model.B_nodes, model.T_set, rule=power_balance_rule)

    model.GenLimits = pyo.Constraint(model.I_set, model.T_set, rule=lambda m, i, t: m.g[i,t] <= m.Pmax[i])
    model.DCPowerFlow = pyo.Constraint(model.L_lines, model.T_set, rule=lambda m, l_from, l_to, t: m.pline[(l_from, l_to),t] == (m.BaseMVA / m.LineReactance[(l_from, l_to)]) * (m.theta[l_from,t] - m.theta[l_to,t]))
    model.LineFlowLimits = pyo.Constraint(model.L_lines, model.T_set, rule=lambda m, l_from, l_to, t: pyo.inequality(-m.LineCapacity[(l_from, l_to)], m.pline[(l_from, l_to), t], m.LineCapacity[(l_from, l_to)]))
    model.ReferenceAngle = pyo.Constraint(model.T_set, rule=lambda m, t: m.theta[m.ReferenceBus, t] == 0.0)
    
    return model

try:
    solver = pyo.SolverFactory('gurobi')
except Exception:
    print("Gurobi not found, falling back to CBC. This may be slow.")
    solver = pyo.SolverFactory('cbc')


# 3 & 4: EXPERIMENTS AND VISUALIZATION

print("\n--- Part 3: Running Simulation Experiments ---")
def get_hourly_demand_at_bus():
    return {bus: [total_demand_profile[t] * nodal_demand_fractions[bus] for t in range(T_horizon)] for bus in nodes_data.keys()}
def run_opf_and_get_results(solar_forecast, emission_params):
    hourly_demand = get_hourly_demand_at_bus()
    model = create_opf_model(hourly_demand, solar_forecast, emission_params)
    results = solver.solve(model, tee=False)
    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        return {'status': str(results.solver.termination_condition), 'cost': np.nan}
    return {'status': 'optimal', 'cost': pyo.value(model.TotalCost)}
target_rmses = np.linspace(10, 250, 15)
mean_solar = true_solar_generation.mean()
bias_levels = np.linspace(-0.50 * mean_solar, 0.50 * mean_solar, 15)
CARBON_PRICE = 50.0
emission_params = {'carbon_price': CARBON_PRICE, 'rates': {gen['id']: gen['Emission_Rate'] for gen in generators_data}}
results_rmse, results_bias = [], []
print("\n--- Running Experiment 1: Varying Forecast Accuracy (RMSE) ---")
for rmse in target_rmses:
    noise = np.random.normal(loc=0, scale=rmse, size=T_horizon)
    noisy_forecast = np.maximum(0, true_solar_generation + noise)
    res = run_opf_and_get_results(noisy_forecast, emission_params)
    res['actual_rmse'] = np.sqrt(mean_squared_error(true_solar_generation, noisy_forecast))
    results_rmse.append(res)
    print(f"Target RMSE: {rmse:6.1f} | Actual RMSE: {res['actual_rmse']:6.1f} | Status: {res['status']:<12} | Cost: ${res.get('cost', 0):,.0f}")
df_rmse = pd.DataFrame(results_rmse)
print("\n--- Running Experiment 2: Varying Forecast Bias ---")
for bias in bias_levels:
    biased_forecast = np.maximum(0, true_solar_generation + bias)
    res = run_opf_and_get_results(biased_forecast, emission_params)
    res['bias_mw'] = bias
    results_bias.append(res)
    print(f"Bias: {bias:6.1f} MW | Status: {res['status']:<12} | Cost: ${res.get('cost', 0):,.0f}")
df_bias = pd.DataFrame(results_bias)
print("\n--- Part 4: Plotting Experiment Results ---")
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(f'Two-Level OPF Experiment Results (Carbon Price: ${CARBON_PRICE}/ton)', fontsize=18, y=1.02)
ax = axes[0]
sns.regplot(x='actual_rmse', y='cost', data=df_rmse.dropna(), ax=ax, line_kws={"color": "red"})
ax.set_title('Experiment 1: Impact of Forecast RMSE on Total Cost', fontsize=14)
ax.set_xlabel('Forecast RMSE (MW)', fontsize=12)
ax.set_ylabel('Total System Cost ($)', fontsize=12)
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
ax = axes[1]
sns.regplot(x='bias_mw', y='cost', data=df_bias.dropna(), ax=ax, line_kws={"color": "red"})
ax.set_title('Experiment 2: Impact of Forecast Bias on Total Cost', fontsize=14)
ax.set_xlabel('Forecast Bias (MW)', fontsize=12)
ax.set_ylabel('Total System Cost ($)', fontsize=12)
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.tight_layout()
plt.show()

print("\nScript finished successfully.")
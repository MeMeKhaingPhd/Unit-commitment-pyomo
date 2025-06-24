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
X.ffill(inplace=True); X.bfill(inplace=True)

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



# 2: OPTIMAL POWER FLOW (OPF) MODEL DEFINITION

print("--- Part 2: Defining the 3-Bus Power System and OPF Model ---")

# System Topology 
BaseMVA = 100.0
nodes_data = {"Bus1": {"is_reference": True}, "Bus2": {}, "Bus3": {}}
# Scale line capacities to handle large real data flows
line_capacity_scaling = 10
lines_data = {
    ("Bus1", "Bus2"): {"reactance_pu": 0.1, "capacity_mw": 200 * line_capacity_scaling},
    ("Bus1", "Bus3"): {"reactance_pu": 0.08, "capacity_mw": 150 * line_capacity_scaling},
    ("Bus2", "Bus3"): {"reactance_pu": 0.05, "capacity_mw": 180 * line_capacity_scaling},
}
print(f"NOTE: Transmission line capacities scaled by {line_capacity_scaling} to be feasible.")
SOLAR_BUS = "Bus2" # Solar is located at Bus 2

# Load and Process Real Data 
try:
    oil_east_data = pd.read_csv(url_oil_east).iloc[0]
    oil_west_data = pd.read_csv(url_oil_west).iloc[0]
    oil_central_data = pd.read_csv(url_oil_central).iloc[0]
    demand_df = pd.read_csv(url_demand_data)
    print("All real data files loaded successfully.")
except Exception as e:
    print(f"\nFATAL ERROR: Could not load a data file from its URL. Error: {e}")
    exit()

# Scale generator capacities to handle large real demand
gen_capacity_scaling = 10
generators_data = [
    {"id": "Oil_East", "bus": "Bus1", "Pmin": 0, "Pmax": oil_east_data['Capacity (MW)'] * gen_capacity_scaling, "Cost_Gen_Linear": 20, "Cost_Gen_Quadratic": 0.02},
    {"id": "Oil_West", "bus": "Bus2", "Pmin": 0, "Pmax": oil_west_data['Capacity (MW)'] * gen_capacity_scaling, "Cost_Gen_Linear": 18, "Cost_Gen_Quadratic": 0.015},
    {"id": "Oil_Central", "bus": "Bus3", "Pmin": 0, "Pmax": oil_central_data['Capacity (MW)'] * gen_capacity_scaling, "Cost_Gen_Linear": 25, "Cost_Gen_Quadratic": 0.03},
]
print(f"NOTE: Conventional generator capacities scaled by {gen_capacity_scaling} to be feasible.")
total_demand_profile = demand_df['Demand (MW)'].iloc[:T_horizon].tolist()
nodal_demand_fractions = {"Bus1": 0.2, "Bus2": 0.5, "Bus3": 0.3}

def create_opf_model(hourly_demand, solar_forecast):
    """Creates a 3-bus Optimal Power Flow (OPF) Pyomo model."""
    model = pyo.ConcreteModel(name="OptimalPowerFlow")
    
    # SETS
    model.I_set = pyo.Set(initialize=[gen['id'] for gen in generators_data])
    model.T_set = pyo.RangeSet(1, T_horizon)
    model.B_nodes = pyo.Set(initialize=nodes_data.keys())
    model.L_lines = pyo.Set(initialize=lines_data.keys(), dimen=2)

    # PARAMETERS
    model.Pmax = pyo.Param(model.I_set, initialize={gen['id']: gen['Pmax'] for gen in generators_data})
    model.Cost_Linear = pyo.Param(model.I_set, initialize={gen['id']: gen['Cost_Gen_Linear'] for gen in generators_data})
    model.Cost_Quad = pyo.Param(model.I_set, initialize={gen['id']: gen['Cost_Gen_Quadratic'] for gen in generators_data})
    model.GenBus = pyo.Param(model.I_set, initialize={gen['id']: gen['bus'] for gen in generators_data}, within=pyo.Any)
    model.LineReactance = pyo.Param(model.L_lines, initialize={line: data['reactance_pu'] for line, data in lines_data.items()})
    model.LineCapacity = pyo.Param(model.L_lines, initialize={line: data['capacity_mw'] for line, data in lines_data.items()})
    model.BaseMVA = pyo.Param(initialize=BaseMVA)
    model.ReferenceBus = pyo.Param(initialize=next(b for b, d in nodes_data.items() if d.get("is_reference")), within=pyo.Any)
    model.SolarPotential = pyo.Param(model.T_set, initialize={t: solar_forecast[t-1] for t in model.T_set})

    # HELPER SETS
    model.GeneratorsAtBus = pyo.Set(model.B_nodes, initialize=lambda m, b: [i for i in m.I_set if m.GenBus[i] == b])
    model.LinesFromBus = pyo.Set(model.B_nodes, initialize=lambda m, b: [l for l in m.L_lines if l[0] == b])
    model.LinesToBus = pyo.Set(model.B_nodes, initialize=lambda m, b: [l for l in m.L_lines if l[1] == b])

    # VARIABLES
    model.g = pyo.Var(model.I_set, model.T_set, domain=pyo.NonNegativeReals)
    model.theta = pyo.Var(model.B_nodes, model.T_set, domain=pyo.Reals)
    model.pline = pyo.Var(model.L_lines, model.T_set, domain=pyo.Reals)
    model.p_curtail = pyo.Var(model.T_set, domain=pyo.NonNegativeReals)

    # OBJECTIVE
    def total_cost_rule(m):
        return sum(m.Cost_Linear[i] * m.g[i, t] + m.Cost_Quad[i] * (m.g[i, t]**2) for i in m.I_set for t in m.T_set)
    model.TotalCost = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

    # CONSTRAINTS
    def power_balance_rule(m, b, t):
        generation_at_bus = sum(m.g[i, t] for i in m.GeneratorsAtBus[b])
        demand_at_bus = hourly_demand[b][t-1]
        solar_injection = 0
        if b == SOLAR_BUS:
            solar_injection = m.SolarPotential[t] - m.p_curtail[t]
        flow_out = sum(m.pline[line, t] for line in m.LinesFromBus[b])
        flow_in = sum(m.pline[line, t] for line in m.LinesToBus[b])
        return generation_at_bus + solar_injection - demand_at_bus == flow_out - flow_in
    model.PowerBalance = pyo.Constraint(model.B_nodes, model.T_set, rule=power_balance_rule)
    
    model.GenLimits = pyo.Constraint(model.I_set, model.T_set, rule=lambda m, i, t: m.g[i, t] <= m.Pmax[i])
    model.CurtailmentLimits = pyo.Constraint(model.T_set, rule=lambda m, t: m.p_curtail[t] <= m.SolarPotential[t])
    model.DCPowerFlow = pyo.Constraint(model.L_lines, model.T_set, rule=lambda m, l_from, l_to, t: m.pline[(l_from, l_to), t] == (m.BaseMVA / m.LineReactance[(l_from, l_to)]) * (m.theta[l_from, t] - m.theta[l_to, t]))
    model.LineFlowLimits = pyo.Constraint(model.L_lines, model.T_set, rule=lambda m, l_from, l_to, t: pyo.inequality(-m.LineCapacity[(l_from, l_to)], m.pline[(l_from, l_to), t], m.LineCapacity[(l_from, l_to)]))
    model.ReferenceAngle = pyo.Constraint(model.T_set, rule=lambda m, t: m.theta[m.ReferenceBus, t] == 0.0)
    
    return model

try:
    solver = pyo.SolverFactory('gurobi')
except Exception:
    print("Gurobi not found, falling back to CBC. This may be slow.")
    solver = pyo.SolverFactory('cbc')



#  3: RUNNING THE TWO-LEVEL SIMULATION

print("\n--- Part 3: Running Two-Level Simulation Experiments ---")

def get_hourly_demand_at_bus():
    """Splits the total demand profile across the three buses."""
    return {bus: [total_demand_profile[t] * nodal_demand_fractions[bus] for t in range(T_horizon)] for bus in nodes_data.keys()}

def run_opf_and_get_results(solar_forecast):
    """Solves the OPF model and returns detailed results."""
    hourly_demand = get_hourly_demand_at_bus()
    model = create_opf_model(hourly_demand, solar_forecast)
    results = solver.solve(model, tee=False)

    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        return {'status': str(results.solver.termination_condition)}

    dispatch = {gen['id']: [pyo.value(model.g[gen['id'], t]) for t in model.T_set] for gen in generators_data}
    dispatch['Solar_Used'] = [pyo.value(model.SolarPotential[t] - model.p_curtail[t]) for t in model.T_set]
    dispatch['Curtailment'] = [pyo.value(model.p_curtail[t]) for t in model.T_set]
    
    return {
        'status': 'optimal',
        'cost': pyo.value(model.TotalCost),
        'dispatch_df': pd.DataFrame(dispatch)
    }

print("\n--- Running Base Case Analysis ---")
base_case_results = run_opf_and_get_results(true_solar_generation)

print("\n--- Running Experiment: Impact of Forecast RMSE on Cost ---")
results_rmse = []
target_rmses = np.linspace(50, 500, 10)
for r in target_rmses:
    noise = np.random.normal(loc=0, scale=r, size=T_horizon)
    noisy_forecast = np.maximum(0, true_solar_generation + noise)
    res = run_opf_and_get_results(noisy_forecast)
    if res.get('status') == 'optimal':
        res['rmse'] = np.sqrt(mean_squared_error(true_solar_generation, noisy_forecast))
        results_rmse.append(res)
        print(f"Target RMSE: {r:6.1f} | Actual RMSE: {res['rmse']:6.1f} | Cost: ${res.get('cost', 0):,.0f}")
df_rmse = pd.DataFrame(results_rmse)

print("\n--- Running Experiment: Impact of Forecast Bias on Cost ---")
results_bias = []
mean_solar = true_solar_generation.mean()
bias_levels = np.linspace(-0.5 * mean_solar, 0.5 * mean_solar, 11)
for bias in bias_levels:
    biased_forecast = np.maximum(0, true_solar_generation + bias)
    res = run_opf_and_get_results(biased_forecast)
    if res.get('status') == 'optimal':
        res['bias'] = bias
        results_bias.append(res)
        print(f"Bias: {bias:6.1f} MW | Status: {res.get('status', 'failed'):<12} | Cost: ${res.get('cost', 0):,.0f}")
df_bias = pd.DataFrame(results_bias)



# 4: VISUALIZATION

print("\n--- Part 4: Visualizing Results ---")
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Two-Level Optimal Power Flow (OPF) Simulation Results', fontsize=18)

# PLOT 1: Dispatch Stack 
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

# PLOT 2: Cost vs. RMSE 
ax2 = axes[0, 1]
if not df_rmse.empty:
    sns.regplot(x='rmse', y='cost', data=df_rmse, ax=ax2, line_kws={"color": "red"})
    ax2.set_title('Impact of Forecast RMSE on Total System Cost', fontsize=14)
    ax2.set_xlabel('Forecast RMSE (MW)')
    ax2.set_ylabel('Total System Cost ($)')
    ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'${format(int(x), ",")}'))
    ax2.grid(True, linestyle='--')

# PLOT 3: Cost vs. Bias 
ax3 = axes[1, 0]
if not df_bias.empty:
    sns.regplot(x='bias', y='cost', data=df_bias, ax=ax3, order=2, line_kws={"color": "purple"}, scatter_kws={'alpha':0.6})
    ax3.set_title('Impact of Forecast Bias on Total System Cost', fontsize=14)
    ax3.set_xlabel('Forecast Bias (MW)')
    ax3.set_ylabel('Total System Cost ($)')
    ax3.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'${format(int(x), ",")}'))
    ax3.grid(True, linestyle='--')

# Turn off the unused subplot 
axes[1, 1].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

print("\nScript finished successfully.")
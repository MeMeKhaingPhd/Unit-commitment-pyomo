import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import seaborn as sns

# PART 1: XGBOOST FORECASTING
print("--- Part 1: Generating Solar Data with XGBoost ---")
try:
    url = 'https://raw.githubusercontent.com/MeMeKhaingPhd/Unit-commitment-pyomo/refs/heads/main/Berlin_solar_regression.csv'
    df = pd.read_csv(url)
    print("\nDataset 'Berlin_solar_regression.csv' loaded successfully from your public source.")
except Exception as e:
    print(f"\nCould not load data from your URL. Error: {e}")
    exit()

df.rename(columns={'X50Hertz..MW.': 'Solar_MW'}, inplace=True)
df['Timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
df = df.set_index('Timestamp').sort_index()

df['hour'] = df.index.hour
df['dayofyear'] = df.index.dayofyear
df['month'] = df.index.month
df['year'] = df.index.year
df['dayofweek'] = df.index.dayofweek

target = 'Solar_MW'
features = [
    'Temperature', 'Clearsky.GHI', 'Cloud.Type', 'Dew.Point', 'GHI', 'DHI',
    'DNI', 'Ozone', 'Relative.Humidity', 'Solar.Zenith.Angle', 'Pressure',
    'Wind.Direction', 'Wind.Speed', 'hour', 'dayofyear', 'month', 'year', 'dayofweek'
]
df.ffill(inplace=True)
df.bfill(inplace=True)

X = df[[f for f in features if f in df.columns]]
y = df[target]

total_demand_profile = [d * 2.5 for d in [100, 110, 105, 100, 120, 150, 180, 220, 250, 260, 270, 280, 275, 260, 240, 220, 200, 180, 160, 150, 140, 130, 120, 110]]
peak_solar_in_data = y.iloc[:int(len(df) * 0.8)].max()
peak_system_demand = max(total_demand_profile)
TARGET_PEAK_SOLAR = peak_system_demand * 0.8
scaling_factor = TARGET_PEAK_SOLAR / peak_solar_in_data if peak_solar_in_data > 0 else 1
y = y * scaling_factor

print(f"\nApplying scaling factor: {scaling_factor:.4f} to solar data.")

split_index = int(len(df) * 0.8)
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]

xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_reg.fit(X_train, y_train)
rmse = np.sqrt(mean_squared_error(y_test, xgb_reg.predict(X_test)))
print(f"Base XGBoost Model Trained. Test RMSE on scaled data: {rmse:.2f} MW")

T = 10 # Using a 10-hour slice for testing
true_solar_generation = y_test.iloc[:T].values
print(f"Using a {T}-hour slice of true (and scaled) solar generation for experiments.\n")


# =============================================================================
# PART 2: OPTIMAL POWER FLOW (OPF) MODEL (Corrected)

BaseMVA = 100.0
T_horizon = T
nodes_data = {"Bus1": {"is_reference": True}, "Bus2": {}, "Bus3": {}}
lines_data = {
    ("Bus1", "Bus2"): {"reactance_pu": 0.1, "capacity_mw": 150},
    ("Bus1", "Bus3"): {"reactance_pu": 0.08, "capacity_mw": 100},
    ("Bus2", "Bus3"): {"reactance_pu": 0.05, "capacity_mw": 120},
}
# generator data with capacity
generators_data = [
    {"id": "G1", "bus": "Bus1", "Pmin": 0, "Pmax": 200, "Cost_Gen_Linear": 10, "Cost_Gen_Quadratic": 0.02, "Emission_Rate": 0.2},
    # Increased Pmax for G2 to ensure system feasibility
    {"id": "G2", "bus": "Bus2", "Pmin": 0, "Pmax": 450, "Cost_Gen_Linear": 15, "Cost_Gen_Quadratic": 0.01, "Emission_Rate": 0.15},
    {"id": "G3", "bus": "Bus3", "Pmin": 0, "Pmax": 100, "Cost_Gen_Linear": 25, "Cost_Gen_Quadratic": 0.05, "Emission_Rate": 0.3},
]
nodal_demand_fractions = {"Bus1": 0.2, "Bus2": 0.5, "Bus3": 0.3}
SOLAR_BUS = "Bus3"

def create_opf_model(hourly_demand, solar_forecast, emission_params=None):
    model = pyo.ConcreteModel(name="OptimalPowerFlow")
    model.I_set = pyo.Set(initialize=[gen['id'] for gen in generators_data])
    model.T_set = pyo.RangeSet(1, T_horizon)
    model.B_nodes = pyo.Set(initialize=nodes_data.keys())
    model.L_lines = pyo.Set(initialize=lines_data.keys(), dimen=2)

    # Parameters 
    model.Pmax = pyo.Param(model.I_set, initialize={gen['id']: gen['Pmax'] for gen in generators_data})
    model.Cost_Gen_Linear = pyo.Param(model.I_set, initialize={gen['id']: gen['Cost_Gen_Linear'] for gen in generators_data})
    model.Cost_Gen_Quadratic = pyo.Param(model.I_set, initialize={gen['id']: gen['Cost_Gen_Quadratic'] for gen in generators_data})
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

    # Decision Variables 
    model.g = pyo.Var(model.I_set, model.T_set, domain=pyo.NonNegativeReals)
    model.theta = pyo.Var(model.B_nodes, model.T_set, domain=pyo.Reals)
    model.pline = pyo.Var(model.L_lines, model.T_set, domain=pyo.Reals)
    model.p_curtail = pyo.Var(model.T_set, domain=pyo.NonNegativeReals, doc="Wasted solar power (MW)")

    # Constraints 
    def max_curtailment_rule(m, t):
        return m.p_curtail[t] <= m.SolarPotential[t]
    model.MaxCurtailment = pyo.Constraint(model.T_set, rule=max_curtailment_rule)
    
    def power_balance_rule(m, b, t):
        generation_at_bus = sum(m.g[i,t] for i in m.GeneratorsAtBus[b])
        demand_at_bus = hourly_demand[b][t-1]
        
        solar_injection = 0
        if b == SOLAR_BUS:
            solar_injection = m.SolarPotential[t] - m.p_curtail[t]
            
        net_demand_at_bus = demand_at_bus - solar_injection
        
        flow_out = sum(m.pline[line,t] for line in m.LinesFromBus[b])
        flow_in = sum(m.pline[line,t] for line in m.LinesToBus[b])
        return generation_at_bus - net_demand_at_bus == flow_out - flow_in
    model.PowerBalance = pyo.Constraint(model.B_nodes, model.T_set, rule=power_balance_rule)

    def total_cost_rule(m):
        gen_cost = sum(m.Cost_Gen_Linear[i] * m.g[i,t] + m.Cost_Gen_Quadratic[i] * (m.g[i,t]**2) for i in m.I_set for t in m.T_set)
        if emission_params:
            emission_cost = emission_params['carbon_price'] * sum(emission_params['rates'][i] * m.g[i,t] for i in m.I_set for t in m.T_set)
            return gen_cost + emission_cost
        return gen_cost
    model.TotalCost = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

    model.GenLimits = pyo.Constraint(model.I_set, model.T_set, rule=lambda m, i, t: m.g[i,t] <= m.Pmax[i])
    def dc_flow_rule(m, l_from, l_to, t):
        line = (l_from, l_to)
        return m.pline[line,t] == (m.BaseMVA / m.LineReactance[line]) * (m.theta[l_from,t] - m.theta[l_to,t])
    model.DCPowerFlow = pyo.Constraint(model.L_lines, model.T_set, rule=dc_flow_rule)
    def line_limit_rule(m, l_from, l_to, t):
        return pyo.inequality(-m.LineCapacity[(l_from, l_to)], m.pline[(l_from, l_to), t], m.LineCapacity[(l_from, l_to)])
    model.LineFlowLimits = pyo.Constraint(model.L_lines, model.T_set, rule=line_limit_rule)
    model.ReferenceAngle = pyo.Constraint(model.T_set, rule=lambda m, t: m.theta[m.ReferenceBus, t] == 0.0)
    return model

try:
    solver = pyo.SolverFactory('gurobi')
except Exception:
    solver = pyo.SolverFactory('cbc')


# =============================================================================
# PART 3: EXPERIMENTS 

def get_base_hourly_demand():
    return {bus: [total_demand_profile[t] * nodal_demand_fractions[bus] for t in range(T_horizon)] for bus in nodes_data.keys()}

def run_opf_and_get_results(solar_forecast, emission_params=None):
    base_demand = get_base_hourly_demand()
    model = create_opf_model(base_demand, solar_forecast, emission_params)
    results = solver.solve(model, tee=False)

    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        return {'status': str(results.solver.termination_condition), 'cost': np.nan, 'total_gen': np.nan, 'total_solar_used': np.nan, 'total_curtailment': np.nan, 'total_emissions': np.nan}

    total_cost = pyo.value(model.TotalCost)
    total_gen = sum(pyo.value(model.g[i,t]) for i in model.I_set for t in model.T_set)
    total_curtailment = sum(pyo.value(model.p_curtail[t]) for t in model.T_set)
    total_solar_used = sum(solar_forecast) - total_curtailment
    
    total_emissions = 0
    if emission_params:
        total_emissions = sum(emission_params['rates'][i] * pyo.value(model.g[i,t]) for i in model.I_set for t in model.T_set)

    return {'status': 'optimal', 'cost': total_cost, 'total_gen': total_gen, 'total_solar_used': total_solar_used, 'total_curtailment': total_curtailment, 'total_emissions': total_emissions}

### Experiments
target_rmses = np.linspace(50, 250, 15)
bias_levels = np.linspace(-0.50 * true_solar_generation.mean(), 0.50 * true_solar_generation.mean(), 15)
penetration_factors = np.linspace(1.0, 3.0, 15)
base_noise = np.random.normal(loc=0, scale=100, size=T_horizon)
base_forecast = np.maximum(0, true_solar_generation + base_noise)
emission_params = {'carbon_price': 50.0, 'rates': {gen['id']: gen['Emission_Rate'] for gen in generators_data}}

results_rmse, results_bias, results_penetration, results_emission = [], [], [], []

print("\n--- Running Experiment 1: Varying Forecast Accuracy (RMSE) ---")
for rmse in target_rmses:
    noisy_fc = np.maximum(0, true_solar_generation + np.random.normal(0, rmse, T_horizon))
    res = run_opf_and_get_results(noisy_fc)
    res['actual_rmse'] = np.sqrt(mean_squared_error(true_solar_generation, noisy_fc))
    results_rmse.append(res)
    print(f"RMSE: {res['actual_rmse']:6.1f} | Status: {res['status']:<12} | Curtailment: {res.get('total_curtailment', 0):6.1f} MW")
df_rmse = pd.DataFrame(results_rmse)

print("\n--- Running Experiment 2: Varying Forecast Bias ---")
for bias in bias_levels:
    biased_fc = np.maximum(0, true_solar_generation + bias)
    res = run_opf_and_get_results(biased_fc)
    res['bias_mw'] = bias
    results_bias.append(res)
    print(f"Bias: {bias:6.1f} MW | Status: {res['status']:<12} | Curtailment: {res.get('total_curtailment', 0):6.1f} MW")
df_bias = pd.DataFrame(results_bias)

print("\n--- Running Experiment 3: Varying Renewable Penetration ---")
for factor in penetration_factors:
    scaled_fc = base_forecast * factor
    res = run_opf_and_get_results(scaled_fc)
    res['penetration_factor'] = factor
    results_penetration.append(res)
    print(f"Factor: {factor:4.2f} | Status: {res['status']:<12} | Curtailment: {res.get('total_curtailment', 0):6.1f} MW")
df_penetration = pd.DataFrame(results_penetration)

print("\n--- Running Experiment 4: Varying Penetration with Emission Costs ---")
for factor in penetration_factors:
    scaled_fc = base_forecast * factor
    res = run_opf_and_get_results(scaled_fc, emission_params=emission_params)
    res['penetration_factor'] = factor
    results_emission.append(res)
    print(f"Factor: {factor:4.2f} | Status: {res['status']:<12} | Curtailment: {res.get('total_curtailment', 0):6.1f} MW")
df_emission = pd.DataFrame(results_emission)


# =============================================================================
# PART 4: VISUALIZING RESULTS
# =============================================================================
print("\n--- Plotting Experiment Results ---")
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('OPF Experiment Results (with Curtailment)', fontsize=16)

# Plot 1: RMSE vs. Cost
ax = axes[0, 0]
sns.regplot(x='actual_rmse', y='cost', data=df_rmse.dropna(), ax=ax, line_kws={"color": "red"}, label='Total Cost')
ax.set_title('Experiment 1: Impact of Forecast RMSE')
ax.set_xlabel('Forecast RMSE (MW)')
ax.set_ylabel('Total System Cost ($)')

# Plot 2: Bias vs. Cost
ax = axes[0, 1]
sns.regplot(x='bias_mw', y='cost', data=df_bias.dropna(), ax=ax, order=2, line_kws={"color": "red"})
ax.set_title('Experiment 2: Impact of Forecast Bias on Total Cost')
ax.set_xlabel('Forecast Bias (MW)')
ax.set_ylabel('Total System Cost ($)')

# Plot 3: Penetration vs. Cost & Curtailment
ax = axes[1, 0]
df_pen_plot = df_penetration.dropna()
line1, = ax.plot(df_pen_plot['penetration_factor'], df_pen_plot['cost'], marker='o', color='blue', label='Total Cost ($)')
ax.set_title('Experiment 3: Impact of Renewable Penetration')
ax.set_xlabel('Penetration Scaling Factor')
ax.set_ylabel('Total System Cost ($)', color='blue')
ax.tick_params(axis='y', labelcolor='blue')
ax2 = ax.twinx()
line2, = ax2.plot(df_pen_plot['penetration_factor'], df_pen_plot['total_curtailment'], marker='x', color='red', linestyle='--', label='Curtailment (MWh)')
ax2.set_ylabel('Total Curtailed Solar (MWh)', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax.legend(handles=[line1, line2])

# Plot 4: Penetration vs. Emissions & Curtailment
ax = axes[1, 1]
df_em_plot = df_emission.dropna()
line1, = ax.plot(df_em_plot['penetration_factor'], df_em_plot['total_emissions'], marker='o', label='Total Emissions', color='green')
ax.set_title('Experiment 4: Penetration vs. Emissions (with Carbon Price)')
ax.set_xlabel('Penetration Scaling Factor')
ax.set_ylabel('Total Emissions (tCO2)', color='green')
ax.tick_params(axis='y', labelcolor='green')
ax2 = ax.twinx()
line2, = ax2.plot(df_em_plot['penetration_factor'], df_em_plot['total_curtailment'], marker='x', color='red', linestyle='--', label='Curtailment (MWh)')
ax2.set_ylabel('Total Curtailed Solar (MWh)', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax.legend(handles=[line1, line2])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("\nScript finished successfully.")
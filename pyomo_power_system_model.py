import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# PART 1: XGBOOST FORECASTING (Corrected and Scaled)
# =============================================================================
print("--- Part 1: Generating Solar Data with XGBoost ---")
try:
    url = 'https://raw.githubusercontent.com/MeMeKhaingPhd/Unit-commitment-pyomo/refs/heads/main/Berlin_solar_regression.csv'
    df = pd.read_csv(url)
    print("\nDataset 'Berlin_solar_regression.csv' loaded successfully from your public source.")
except Exception as e:
    print(f"\nCould not load data from your URL. Error: {e}")
    print("Please check that the URL is correct and the repository is public.")
    exit()

# Preprocessing and Feature Engineering
df.rename(columns={'X50Hertz..MW.': 'Solar_MW'}, inplace=True)
df['Timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
df = df.set_index('Timestamp')
df.sort_index(inplace=True)

df['hour'] = df.index.hour
df['dayofyear'] = df.index.dayofyear
df['month'] = df.index.month
df['year'] = df.index.year
df['dayofweek'] = df.index.dayofweek

target = 'Solar_MW'
features = [
    'Temperature', 'Clearsky.GHI', 'Cloud.Type', 'Dew.Point', 'GHI', 'DHI',
    'DNI', 'Ozone', 'Relative.Humidity', 'Solar.Zenith.Angle', 'Pressure',
    'Wind.Direction', 'Wind.Speed',
    'hour', 'dayofyear', 'month', 'year', 'dayofweek'
]
existing_features = [f for f in features if f in df.columns]
if len(existing_features) != len(features):
    print("\nWarning: Some specified features were not found in the DataFrame.")
    features = existing_features

# FIX: Use modern .ffill()/.bfill() to avoid FutureWarnings
df.ffill(inplace=True)
df.bfill(inplace=True)

X = df[features]
y = df[target]

print(f"\nFeatures selected for the model: {features}")

# --- DYNAMIC DATA SCALING TO PREVENT INFEASIBILITY ---
# This is the primary fix for the solver errors.
# We define a placeholder demand profile here so we can access it for scaling.
total_demand_profile = [d * 2.5 for d in [100, 110, 105, 100, 120, 150, 180, 220, 250, 260, 270, 280, 275, 260, 240, 220, 200, 180, 160, 150, 140, 130, 120, 110]]

# Find the peak of the training data to create a scaling factor
peak_solar_in_data = y.iloc[:int(len(df) * 0.8)].max()
peak_system_demand = max(total_demand_profile)
# Scale the solar data so its peak is about 60% of the system's peak demand
TARGET_PEAK_SOLAR = peak_system_demand * 0.6
scaling_factor = TARGET_PEAK_SOLAR / peak_solar_in_data if peak_solar_in_data > 0 else 1

print(f"\nOriginal peak solar in training data: {peak_solar_in_data:.2f} MW")
print(f"Target peak solar (for model stability): {TARGET_PEAK_SOLAR:.2f} MW")
print(f"Applying scaling factor: {scaling_factor:.4f}")

# Apply the scaling factor to the entire target variable
y = y * scaling_factor
# --- END SCALING STEP ---

# Time-series split (80% train, 20% test)
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Model Training
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_reg.fit(X_train, y_train)
y_pred = xgb_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Base XGBoost Model Trained. Test RMSE on scaled data: {rmse:.2f} MW")

T = 5# Use a 5 hour horizon for more meaningful results
true_solar_generation = y_test.iloc[:T].values
print(f"Using a {T}-hour slice of true (and scaled) solar generation for experiments.\n")


# =============================================================================
# PART 2: OPTIMAL POWER FLOW (OPF) MODEL 

BaseMVA = 100.0
T_horizon = T

nodes_data = {"Bus1": {"is_reference": True}, "Bus2": {}, "Bus3": {}}
lines_data = {
    ("Bus1", "Bus2"): {"reactance_pu": 0.1, "capacity_mw": 150},
    ("Bus1", "Bus3"): {"reactance_pu": 0.08, "capacity_mw": 100},
    ("Bus2", "Bus3"): {"reactance_pu": 0.05, "capacity_mw": 120},
}
generators_data = [
    {"id": "G1", "bus": "Bus1", "Pmin": 0, "Pmax": 200, "Cost_Gen_Linear": 10, "Cost_Gen_Quadratic": 0.02, "Emission_Rate": 0.2},
    {"id": "G2", "bus": "Bus2", "Pmin": 0, "Pmax": 300, "Cost_Gen_Linear": 15, "Cost_Gen_Quadratic": 0.01, "Emission_Rate": 0.15},
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

    model.Pmax = pyo.Param(model.I_set, initialize={gen['id']: gen['Pmax'] for gen in generators_data})
    model.Cost_Gen_Linear = pyo.Param(model.I_set, initialize={gen['id']: gen['Cost_Gen_Linear'] for gen in generators_data})
    model.Cost_Gen_Quadratic = pyo.Param(model.I_set, initialize={gen['id']: gen['Cost_Gen_Quadratic'] for gen in generators_data})
    # FIX: Add 'within=pyo.Any' to suppress deprecation warnings
    model.GenBus = pyo.Param(model.I_set, initialize={gen['id']: gen['bus'] for gen in generators_data}, within=pyo.Any)
    model.LineReactance = pyo.Param(model.L_lines, initialize={line: data['reactance_pu'] for line, data in lines_data.items()})
    model.LineCapacity = pyo.Param(model.L_lines, initialize={line: data['capacity_mw'] for line, data in lines_data.items()})
    model.BaseMVA = pyo.Param(initialize=BaseMVA)
    # FIX: Add 'within=pyo.Any' to suppress deprecation warnings
    model.ReferenceBus = pyo.Param(initialize=next(b for b, d in nodes_data.items() if d.get("is_reference")), within=pyo.Any)

    def net_demand_rule(m, b, t):
        base_demand = hourly_demand[b][t-1]
        solar_gen = solar_forecast[t-1] if b == SOLAR_BUS else 0
        return base_demand - solar_gen
    model.NetDemand = pyo.Param(model.B_nodes, model.T_set, rule=net_demand_rule)

    model.GeneratorsAtBus = pyo.Set(model.B_nodes, initialize=lambda m, b: [i for i in m.I_set if m.GenBus[i] == b])
    model.LinesFromBus = pyo.Set(model.B_nodes, initialize=lambda m, b: [l for l in m.L_lines if l[0] == b])
    model.LinesToBus = pyo.Set(model.B_nodes, initialize=lambda m, b: [l for l in m.L_lines if l[1] == b])

    model.g = pyo.Var(model.I_set, model.T_set, domain=pyo.NonNegativeReals, doc="Generation level (MW)")
    model.theta = pyo.Var(model.B_nodes, model.T_set, domain=pyo.Reals, doc="Bus voltage angle (radians)")
    model.pline = pyo.Var(model.L_lines, model.T_set, domain=pyo.Reals, doc="Power flow on lines (MW)")

    def total_cost_rule(m):
        gen_cost = sum(m.Cost_Gen_Linear[i] * m.g[i,t] + m.Cost_Gen_Quadratic[i] * (m.g[i,t]**2) for i in m.I_set for t in m.T_set)
        if emission_params:
            emission_cost = emission_params['carbon_price'] * sum(emission_params['rates'][i] * m.g[i,t] for i in m.I_set for t in m.T_set)
            return gen_cost + emission_cost
        return gen_cost
    model.TotalCost = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

    model.GenLimits = pyo.Constraint(model.I_set, model.T_set, rule=lambda m, i, t: m.g[i,t] <= m.Pmax[i])
    def power_balance_rule(m, b, t):
        generation_at_bus = sum(m.g[i,t] for i in m.GeneratorsAtBus[b])
        net_demand_at_bus = m.NetDemand[b,t]
        flow_out = sum(m.pline[line,t] for line in m.LinesFromBus[b])
        flow_in = sum(m.pline[line,t] for line in m.LinesToBus[b])
        return generation_at_bus - net_demand_at_bus == flow_out - flow_in
    model.PowerBalance = pyo.Constraint(model.B_nodes, model.T_set, rule=power_balance_rule)
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
    print("Gurobi not found, trying CBC (included with Pyomo)")
    solver = pyo.SolverFactory('cbc')


# =============================================================================
# PART 3: EXPERIMENTS 

def get_base_hourly_demand():
    return {bus: [total_demand_profile[t] * nodal_demand_fractions[bus] for t in range(T_horizon)] for bus in nodes_data.keys()}

# FIX: Rewritten function to be robust to solver failures
def run_opf_and_get_results(solar_forecast, emission_params=None):
    base_demand = get_base_hourly_demand()
    model = create_opf_model(base_demand, solar_forecast, emission_params)
    results = solver.solve(model, tee=False)

    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        # Return a dictionary with NaN values to avoid KeyErrors and allow plotting
        return {
            'status': str(results.solver.termination_condition),
            'cost': np.nan,
            'total_gen': np.nan,
            'total_solar': np.nan,
            'total_emissions': np.nan
        }

    total_cost = pyo.value(model.TotalCost)
    total_gen = sum(pyo.value(model.g[i,t]) for i in model.I_set for t in model.T_set)
    total_solar = sum(solar_forecast)
    total_emissions = 0
    if emission_params:
        total_emissions = sum(emission_params['rates'][i] * pyo.value(model.g[i,t]) for i in model.I_set for t in model.T_set)

    return {
        'status': 'optimal',
        'cost': total_cost,
        'total_gen': total_gen,
        'total_solar': total_solar,
        'total_emissions': total_emissions
    }

# Experiment 1: Varying Forecast Accuracy (RMSE)
print("\n--- Running Experiment 1: Varying Forecast Accuracy (RMSE) ---")
results_rmse = []
# Adjusted RMSE range to be more realistic for the scaled data
target_rmses = np.linspace(50, 250, 15)

for target_rmse in target_rmses:
    noise = np.random.normal(loc=0, scale=target_rmse, size=T_horizon)
    noisy_forecast = np.maximum(0, true_solar_generation + noise)
    actual_rmse = np.sqrt(mean_squared_error(true_solar_generation, noisy_forecast))
    result = run_opf_and_get_results(noisy_forecast)
    results_rmse.append({'target_rmse': target_rmse, 'actual_rmse': actual_rmse, 'total_cost': result['cost']})
    print(f"Target RMSE: {target_rmse:6.1f} | Actual RMSE: {actual_rmse:6.1f} | Status: {result['status']:<12} | Cost: ${result['cost']:,.0f}")
df_rmse = pd.DataFrame(results_rmse)

#Experiment 2: Varying Forecast Bias 
print("\n--- Running Experiment 2: Varying Forecast Bias ---")
results_bias = []
mean_solar = true_solar_generation.mean()
bias_levels = np.linspace(-0.50 * mean_solar, 0.50 * mean_solar, 15)

for bias in bias_levels:
    biased_forecast = np.maximum(0, true_solar_generation + bias)
    result = run_opf_and_get_results(biased_forecast)
    results_bias.append({'bias_mw': bias, 'total_cost': result['cost']})
    print(f"Forecast Bias: {bias:6.1f} MW | Status: {result['status']:<12} | Cost: ${result['cost']:,.0f}")
df_bias = pd.DataFrame(results_bias)

#Experiment 3: Varying Renewable Penetration 
print("\n--- Running Experiment 3: Varying Renewable Penetration ---")
results_penetration = []
penetration_factors = np.linspace(1.0, 2.5, 15)
fixed_rmse_for_penetration = 100 # Scaled down RMSE

base_noise = np.random.normal(loc=0, scale=fixed_rmse_for_penetration, size=T_horizon)
base_forecast = np.maximum(0, true_solar_generation + base_noise)

for factor in penetration_factors:
    scaled_solar_forecast = base_forecast * factor
    result = run_opf_and_get_results(scaled_solar_forecast)
    results_penetration.append({
        'penetration_factor': factor, 'total_cost': result['cost'],
        'total_solar_gen': result['total_solar'], 'total_thermal_gen': result['total_gen']
    })
    print(f"Penetration Factor: {factor:4.2f} | Status: {result['status']:<12} | Cost: ${result['cost']:,.0f}")
df_penetration = pd.DataFrame(results_penetration)

#Experiment 4: Varying Penetration with Emission Costs
print("\n--- Running Experiment 4: Varying Penetration with Emission Costs ---")
results_emission = []
emission_params = {'carbon_price': 50.0, 'rates': {gen['id']: gen['Emission_Rate'] for gen in generators_data}}

for factor in penetration_factors:
    scaled_solar_forecast = base_forecast * factor
    result = run_opf_and_get_results(scaled_solar_forecast, emission_params=emission_params)
    results_emission.append({
        'penetration_factor': factor, 'total_cost': result['cost'],
        'total_solar_gen': result['total_solar'], 'total_thermal_gen': result['total_gen'],
        'total_emissions_tco2': result['total_emissions']
    })
    print(f"Penetration Factor: {factor:4.2f} | Status: {result['status']:<12} | Emissions: {result['total_emissions']:6.0f} tCO2 | Cost: ${result['cost']:,.0f}")
df_emission = pd.DataFrame(results_emission)


# =============================================================================
# PART 4: VISUALIZING RESULTS

print("\n--- Plotting Experiment Results ---")
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('OPF Experiment Results', fontsize=16)

# Plot 1: RMSE vs. Cost
sns.regplot(x='actual_rmse', y='total_cost', data=df_rmse.dropna(), ax=axes[0, 0], line_kws={"color": "red"})
axes[0, 0].set_title('Experiment 1: Impact of Forecast RMSE on Total Cost')
axes[0, 0].set_xlabel('Forecast RMSE (MW)')
axes[0, 0].set_ylabel('Total System Cost ($)')

# Plot 2: Bias vs. Cost
sns.regplot(x='bias_mw', y='total_cost', data=df_bias.dropna(), ax=axes[0, 1], order=2, line_kws={"color": "red"})
axes[0, 1].set_title('Experiment 2: Impact of Forecast Bias on Total Cost')
axes[0, 1].set_xlabel('Forecast Bias (MW)')
axes[0, 1].set_ylabel('Total System Cost ($)')

# Plot 3: Penetration vs. Cost
ax3 = axes[1, 0]
df_pen_plot = df_penetration.dropna()
ax3.plot(df_pen_plot['penetration_factor'], df_pen_plot['total_cost'], marker='o', label='Total Cost')
ax3.set_title('Experiment 3: Impact of Renewable Penetration')
ax3.set_xlabel('Penetration Scaling Factor')
ax3.set_ylabel('Total System Cost ($)', color='blue')
ax3_twin = ax3.twinx()
ax3_twin.plot(df_pen_plot['penetration_factor'], df_pen_plot['total_thermal_gen'], marker='x', color='green', linestyle='--', label='Thermal Generation')
ax3_twin.set_ylabel('Total Thermal Generation (MWh)', color='green')
# Manually create legends for twin axes plots
lines, labels = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3_twin.legend(lines + lines2, labels + labels2, loc='upper left')

# Plot 4: Penetration vs. Emissions
ax4 = axes[1, 1]
df_em_plot = df_emission.dropna()
ax4.plot(df_em_plot['penetration_factor'], df_em_plot['total_emissions_tco2'], marker='o', label='Total Emissions', color='red')
ax4.set_title('Experiment 4: Penetration vs. Emissions (with Carbon Price)')
ax4.set_xlabel('Penetration Scaling Factor')
ax4.set_ylabel('Total Emissions (tCO2)', color='red')
ax4_twin = ax4.twinx()
ax4_twin.plot(df_em_plot['penetration_factor'], df_em_plot['total_cost'], marker='x', color='purple', linestyle='--', label='Total Cost (incl. carbon)')
ax4_twin.set_ylabel('Total System Cost ($)', color='purple')
lines, labels = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4_twin.legend(lines + lines2, labels + labels2, loc='upper left')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("\nScript finished successfully.")
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# PART 0: CONFIGURING DATA SOURCES
# =============================================================================
print("--- Part 0: Configuring Data Sources ---")
cleaned_solar_data_file = 'https://raw.githubusercontent.com/MeMeKhaingPhd/Unit-commitment-pyomo/refs/heads/main/solar_data_cleaned.csv'
url_demand_data = 'https://raw.githubusercontent.com/MeMeKhaingPhd/Unit-commitment-pyomo/refs/heads/main/dem-data-berlin.csv'
url_oil_east = 'https://raw.githubusercontent.com/MeMeKhaingPhd/Unit-commitment-pyomo/refs/heads/main/oil-data-east-berlin.csv'
url_oil_west = 'https://raw.githubusercontent.com/MeMeKhaingPhd/Unit-commitment-pyomo/refs/heads/main/oil-data-west-berlin.csv'
url_oil_central ='https://raw.githubusercontent.com/MeMeKhaingPhd/Unit-commitment-pyomo/refs/heads/main/oil-data-central-berlin.csv'

# =============================================================================
# PART 1: SOLAR FORECASTING MODEL SETUP
# =============================================================================
print("\n--- Part 1: Training the Solar Forecasting Model ---")
try:
    df = pd.read_csv(cleaned_solar_data_file)
    print("Solar data loaded successfully from URL.")
except Exception as e:
    print(f"\nFATAL ERROR: Could not load solar data from the URL. Error: {e}")
    exit()

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


# =============================================================================
# PART 2: 3-BUS POWER SYSTEM (OPF) MODEL DEFINITION
# =============================================================================
print("--- Part 2: Defining the 3-Bus Power System and OPF Model ---")
# (This section remains unchanged as it's highly specific and not repetitive)
BaseMVA = 100.0
nodes_data = {"Bus1": {"is_reference": True}, "Bus2": {}, "Bus3": {}}
line_capacity_scaling = 10
lines_data = {
    ("Bus1", "Bus2"): {"reactance_pu": 0.1, "capacity_mw": 200 * line_capacity_scaling},
    ("Bus1", "Bus3"): {"reactance_pu": 0.08, "capacity_mw": 150 * line_capacity_scaling},
    ("Bus2", "Bus3"): {"reactance_pu": 0.05, "capacity_mw": 180 * line_capacity_scaling},
}
SOLAR_BUS = "Bus2"
try:
    oil_east_data = pd.read_csv(url_oil_east).iloc[0]
    oil_west_data = pd.read_csv(url_oil_west).iloc[0]
    oil_central_data = pd.read_csv(url_oil_central).iloc[0]
    demand_df = pd.read_csv(url_demand_data)
    print("All real data files loaded successfully.")
except Exception as e:
    print(f"\nFATAL ERROR: Could not load a data file from its URL. Error: {e}")
    exit()
gen_capacity_scaling = 10
generators_data = [
    {"id": "Oil_East", "bus": "Bus1", "Pmin": 0, "Pmax": oil_east_data['Capacity (MW)'] * gen_capacity_scaling, "Cost_Gen_Linear": 20, "Cost_Gen_Quadratic": 0.02},
    {"id": "Oil_West", "bus": "Bus2", "Pmin": 0, "Pmax": oil_west_data['Capacity (MW)'] * gen_capacity_scaling, "Cost_Gen_Linear": 18, "Cost_Gen_Quadratic": 0.015},
    {"id": "Oil_Central", "bus": "Bus3", "Pmin": 0, "Pmax": oil_central_data['Capacity (MW)'] * gen_capacity_scaling, "Cost_Gen_Linear": 25, "Cost_Gen_Quadratic": 0.03},
]
total_demand_profile = demand_df['Demand (MW)'].iloc[:T_horizon].tolist()
nodal_demand_fractions = {"Bus1": 0.2, "Bus2": 0.5, "Bus3": 0.3}

def create_opf_model(hourly_demand_at_bus, solar_forecast):
    model = pyo.ConcreteModel(name="OptimalPowerFlow")
    model.I_set = pyo.Set(initialize=[gen['id'] for gen in generators_data])
    model.T_set = pyo.RangeSet(1, T_horizon)
    model.B_nodes = pyo.Set(initialize=nodes_data.keys())
    model.L_lines = pyo.Set(initialize=lines_data.keys(), dimen=2)
    model.Pmax = pyo.Param(model.I_set, initialize={gen['id']: gen['Pmax'] for gen in generators_data})
    model.Cost_Linear = pyo.Param(model.I_set, initialize={gen['id']: gen['Cost_Gen_Linear'] for gen in generators_data})
    model.Cost_Quad = pyo.Param(model.I_set, initialize={gen['id']: gen['Cost_Gen_Quadratic'] for gen in generators_data})
    model.GenBus = pyo.Param(model.I_set, initialize={gen['id']: gen['bus'] for gen in generators_data}, within=pyo.Any)
    model.LineReactance = pyo.Param(model.L_lines, initialize={line: data['reactance_pu'] for line, data in lines_data.items()})
    model.LineCapacity = pyo.Param(model.L_lines, initialize={line: data['capacity_mw'] for line, data in lines_data.items()})
    model.BaseMVA = pyo.Param(initialize=BaseMVA)
    model.ReferenceBus = pyo.Param(initialize=next(b for b, d in nodes_data.items() if d.get("is_reference")), within=pyo.Any)
    model.SolarPotential = pyo.Param(model.T_set, initialize={t: solar_forecast[t-1] for t in model.T_set})
    model.GeneratorsAtBus = pyo.Set(model.B_nodes, initialize=lambda m, b: [i for i in m.I_set if m.GenBus[i] == b])
    model.LinesFromBus = pyo.Set(model.B_nodes, initialize=lambda m, b: [l for l in m.L_lines if l[0] == b])
    model.LinesToBus = pyo.Set(model.B_nodes, initialize=lambda m, b: [l for l in m.L_lines if l[1] == b])
    model.g = pyo.Var(model.I_set, model.T_set, domain=pyo.NonNegativeReals)
    model.theta = pyo.Var(model.B_nodes, model.T_set, domain=pyo.Reals)
    model.pline = pyo.Var(model.L_lines, model.T_set, domain=pyo.Reals)
    model.p_curtail = pyo.Var(model.T_set, domain=pyo.NonNegativeReals)
    model.TotalCost = pyo.Objective(rule=lambda m: sum(m.Cost_Linear[i] * m.g[i, t] + m.Cost_Quad[i] * (m.g[i, t]**2) for i in m.I_set for t in m.T_set), sense=pyo.minimize)
    model.PowerBalance = pyo.Constraint(model.B_nodes, model.T_set, rule=lambda m, b, t: sum(m.g[i, t] for i in m.GeneratorsAtBus[b]) + (m.SolarPotential[t] - m.p_curtail[t] if b == SOLAR_BUS else 0) - hourly_demand_at_bus[b][t-1] == sum(m.pline[l, t] for l in m.LinesFromBus[b]) - sum(m.pline[l, t] for l in m.LinesToBus[b]))
    model.GenLimits = pyo.Constraint(model.I_set, model.T_set, rule=lambda m, i, t: m.g[i, t] <= m.Pmax[i])
    model.CurtailmentLimits = pyo.Constraint(model.T_set, rule=lambda m, t: m.p_curtail[t] <= m.SolarPotential[t])
    model.DCPowerFlow = pyo.Constraint(model.L_lines, model.T_set, rule=lambda m, l_from, l_to, t: m.pline[(l_from, l_to), t] == (m.BaseMVA / m.LineReactance[(l_from, l_to)]) * (m.theta[l_from, t] - m.theta[l_to, t]))
    model.LineFlowLimits = pyo.Constraint(model.L_lines, model.T_set, rule=lambda m, l_from, l_to, t: pyo.inequality(-m.LineCapacity[(l_from, l_to)], m.pline[(l_from, l_to), t], m.LineCapacity[(l_from, l_to)]))
    model.ReferenceAngle = pyo.Constraint(model.T_set, rule=lambda m, t: m.theta[m.ReferenceBus, t] == 0.0)
    return model

try:
    solver = pyo.SolverFactory('gurobi')
except Exception:
    print("Gurobi not found, falling back to CBC.")
    solver = pyo.SolverFactory('cbc')


# =============================================================================
# PART 3: SIMULATION EXPERIMENTS (DATA GENERATION)
# =============================================================================
np.random.seed(42)
print("\n--- Part 3: Running All Simulation Experiments ---")

def get_hourly_demand_at_bus():
    return {bus: [total_demand_profile[t] * nodal_demand_fractions[bus] for t in range(T_horizon)] for bus in nodes_data.keys()}

def run_opf_and_get_results(solar_forecast):
    hourly_demand = get_hourly_demand_at_bus()
    model = create_opf_model(hourly_demand, solar_forecast)
    results = solver.solve(model, tee=False)
    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        return {'status': str(results.solver.termination_condition), 'cost': np.nan}
    dispatch = {gen['id']: [pyo.value(model.g[gen['id'], t]) for t in model.T_set] for gen in generators_data}
    curtailment = [pyo.value(model.p_curtail[t]) for t in model.T_set]
    return {'status': 'optimal', 'cost': pyo.value(model.TotalCost), 'dispatch': dispatch, 'curtailment': curtailment}

# --- Experiment 3A: Calculating Composite Loss ---
print("\n--- Experiment 3A: Calculating Composite Loss from Accuracy Variations ---")
PENALTY_RHO = 5000
CURTAILMENT_COST_PER_MWH = 20
results_composite_loss = []
target_rmses = np.linspace(50, 500, 15)
for r in target_rmses:
    noise = np.random.normal(loc=0, scale=r, size=T_horizon)
    dispatch_forecast = np.maximum(0, true_solar_generation + noise)
    dispatch_results = run_opf_and_get_results(dispatch_forecast)
    if dispatch_results.get('status') == 'optimal':
        demand_t = np.array(total_demand_profile)
        reserve_t = np.sum([dispatch_results['dispatch'][g['id']] for g in generators_data], axis=0)
        shortfall_amount_t = np.maximum(0, demand_t - reserve_t - true_solar_generation)
        total_composite_loss = np.sum(PENALTY_RHO * shortfall_amount_t) # Simplified for this example
        results_composite_loss.append({
            'rmse': np.sqrt(mean_squared_error(true_solar_generation, dispatch_forecast)),
            'composite_loss': total_composite_loss
        })
df_composite_loss = pd.DataFrame(results_composite_loss)
print("Finished composite loss experiment.")


# --- Experiment 3B: Asymmetric Accuracy (Varying rho) ---
print("\n--- Experiment 3B: Sensitivity Analysis of Asymmetric Penalty (rho) ---")
def wmse_asymmetric_loss(rho):
    def custom_loss(preds, dtrain):
        labels = dtrain.get_label()
        residual = preds - labels
        grad = np.where(residual >= 0, 2 * rho * residual, 2 * (1 - rho) * residual)
        hess = np.where(residual >= 0, 2 * rho, 2 * (1 - rho))
        return grad, hess
    return custom_loss

rho_steps = np.arange(0.75, 1.01, 0.05)
results_asymmetric = []
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
watchlist = [(dtrain, 'train'), (dtest, 'eval')]
params = {'learning_rate': 0.05, 'max_depth': 5, 'random_state': 42, 'n_jobs': -1}

for rho_value in rho_steps:
    name = f"rho={rho_value:.2f}, kappa={1-rho_value:.2f}"
    print(f"Training with Asymmetric Loss: {name}...")
    bst = xgb.train(params=params, dtrain=dtrain, num_boost_round=1000, evals=watchlist, obj=wmse_asymmetric_loss(rho_value), early_stopping_rounds=50, verbose_eval=False)
    predictions = bst.predict(dtest)
    dispatch_forecast = np.maximum(0, predictions[:T_horizon])
    dispatch_results = run_opf_and_get_results(dispatch_forecast)
    if dispatch_results.get('status') == 'optimal':
        results_asymmetric.append({'Model Type': name, 'rho': rho_value, 'Cost': dispatch_results['cost'], 'RMSE': np.sqrt(mean_squared_error(y_test, predictions)), 'MBE': np.mean(predictions - y_test)})
df_asymmetric = pd.DataFrame(results_asymmetric)
print("Finished varying rho experiment.")


# --- Experiment 3C: Standard Loss Comparison ---
print("\n--- Experiment 3C: Comparing Standard ML Losses (MSE, MAE, etc.) ---")
standard_losses_to_test = {'Standard MSE (L2)': 'reg:squarederror', 'Standard MAE (L1)': 'reg:absoluteerror', 'Standard Huber': 'reg:pseudohubererror'}
results_standard_loss = []
for name, objective_func in standard_losses_to_test.items():
    print(f"Training XGBoost model with Standard Loss: {name}...")
    model = xgb.XGBRegressor(objective=objective_func, n_estimators=1000, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    predictions = model.predict(X_test)
    dispatch_forecast = np.maximum(0, predictions[:T_horizon])
    dispatch_results = run_opf_and_get_results(dispatch_forecast)
    if dispatch_results.get('status') == 'optimal':
        results_standard_loss.append({'Model Type': name, 'Cost': dispatch_results['cost'], 'RMSE': np.sqrt(mean_squared_error(y_test, predictions)), 'MBE': np.mean(predictions - y_test)})
df_standard_loss = pd.DataFrame(results_standard_loss)
print("Finished standard loss comparison experiment.")

df_all_models = pd.concat([df_asymmetric, df_standard_loss], ignore_index=True)


# =============================================================================
# PART 4: NUMERICAL RESULTS AND CSV EXPORT
# =============================================================================
print("\n" + "="*80)
print("--- FINAL NUMERICAL RESULTS FOR REPORT ---")
print("="*80)
if not df_composite_loss.empty:
    print("\n--- Numerical Results for Composite Loss (RMSE Experiment) ---")
    print(df_composite_loss[['rmse', 'composite_loss']].round(2))
if not df_all_models.empty:
    print("\n--- Numerical Results for Asymmetric & Standard Objectives (Ranked by Cost) ---")
    print(df_all_models[['Model Type', 'Cost', 'RMSE', 'MBE']].sort_values('Cost').round(2))
print("\n" + "="*80)
print("\n--- EXPORTING FINAL RESULTS TO CSV FILES ---")
try:
    if not df_composite_loss.empty: df_composite_loss.round(2).to_csv('results_composite_loss.csv', index=False)
    if not df_all_models.empty: df_all_models.sort_values('Cost').round(2).to_csv('results_model_comparison.csv', index=False)
    print("Successfully saved numerical results to CSV files.")
except Exception as e:
    print(f"\nERROR: Could not save results to CSV. Error: {e}")
print("\n" + "="*80)


# =============================================================================
# PART 5: VISUALIZATION FUNCTIONS (REFACTORED)
# =============================================================================
print("\n--- Part 5: Generating All Visualizations ---")

# --- Centralized Styling ---
TITLE_FONT = {'size': 26, 'weight': 'bold'}
LABEL_FONT = {'size': 22, 'weight': 'bold'}
TICK_FONT_SIZE = 18
sns.set_style("whitegrid", {'grid.linestyle': '--'})

def plot_composite_loss_vs_rmse(df, save_path):
    print(f"Generating: {save_path}")
    fig, ax = plt.subplots(figsize=(16, 9))
    sns.regplot(x='rmse', y='composite_loss', data=df, ax=ax, color='teal', scatter_kws={'alpha': 0.9, 's': 150})
    ax.set_title('Composite Loss vs. Forecast Inaccuracy', fontdict=TITLE_FONT)
    ax.set_xlabel('Forecast RMSE (MW)', fontdict=LABEL_FONT)
    ax.set_ylabel('Composite Loss ($)', fontdict=LABEL_FONT)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: f'${format(int(x), ",")}'))
    ax.axhline(0, color='black', lw=1.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_performance_scatter_with_inset(df, save_path):
    print(f"Generating: {save_path}")
    plt.rcParams.update({'font.weight': 'bold', 'axes.linewidth': 2.5})
    fig, ax = plt.subplots(figsize=(16, 10))
    model_markers = ["o", "X", "s", "P", "D", "D", "^", "s", "v"]
    palette = sns.color_palette("viridis", n_colors=len(df))
    sns.scatterplot(x='RMSE', y='Cost', data=df, hue='Model Type', style='Model Type', markers=model_markers, palette=palette, s=600, ax=ax, edgecolor='black', linewidth=2)
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    ax_inset = ax.inset_axes([0.45, 0.4, 0.4, 0.5])
    sns.scatterplot(x='RMSE', y='Cost', data=df, hue='Model Type', style='Model Type', markers=model_markers, palette=palette, s=600, ax=ax_inset, edgecolor='black', linewidth=2, legend=False)
    ax_inset.set_xlim(215, 280); ax_inset.set_ylim(1500, 6500)
    ax_inset.tick_params(axis='both', which='major', labelsize=14, width=2.5)
    ax.indicate_inset_zoom(ax_inset, edgecolor="black", alpha=1, lw=2.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_title('Performance of Models Trained with Different Objectives', fontdict=TITLE_FONT, loc='left')
    ax.set_xlabel('Resulting Forecast RMSE (MW)', fontdict=LABEL_FONT)
    ax.set_ylabel('Resulting System Cost ($)', fontdict=LABEL_FONT)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: f'${format(int(x), ",")}'))
    legend = fig.legend(handles=handles, labels=labels, title='Training Objective', bbox_to_anchor=(0.99, 1.0), loc='upper right', fontsize=16, title_fontsize=18, frameon=True, edgecolor='black')
    legend.get_frame().set_linewidth(2.5); legend.get_title().set_fontweight('bold')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.rcParams.update(plt.rcParamsDefault) # Reset style

def plot_economic_ranking(df, save_path, log_scale=False):
    print(f"Generating: {save_path}")
    fig, ax = plt.subplots(figsize=(16, 10))
    df_sorted = df.sort_values('Cost', ascending=True)
    sns.barplot(x='Cost', y='Model Type', data=df_sorted, ax=ax, palette='plasma')
    title = 'Economic Performance Ranking of Models'
    xlabel = 'Total System Cost ($)'
    if log_scale:
        ax.set_xscale('log')
        title += ' (Log Scale)'
        xlabel += ' [Log Scale]'
    ax.set_title(title, fontdict=TITLE_FONT)
    ax.set_xlabel(xlabel, fontdict=LABEL_FONT)
    ax.set_ylabel('Training Objective', fontdict=LABEL_FONT)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: f'${format(int(x), ",")}'))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_forecast_bias(df, save_path, symlog_scale=False):
    print(f"Generating: {save_path}")
    fig, ax = plt.subplots(figsize=(18, 10))
    df_sorted = df.sort_values('Cost', ascending=True)
    sns.barplot(x='MBE', y='Model Type', data=df_sorted, ax=ax, palette='plasma')
    title = 'Resulting Forecast Bias (MBE) of Models'
    xlabel = 'Mean Bias Error (MW)'
    if symlog_scale:
        ax.set_xscale('symlog', linthresh=50)
        title += ' (Symlog Scale)'
        xlabel += ' [Symlog Scale]'
    ax.set_title(title, fontdict=TITLE_FONT)
    ax.set_xlabel(xlabel, fontdict=LABEL_FONT)
    ax.set_ylabel('')
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_rho_sensitivity(df, save_path, log_scale=False):
    print(f"Generating: {save_path}")
    fig, ax = plt.subplots(figsize=(16, 9))
    sns.lineplot(x='rho', y='Cost', data=df, ax=ax, marker='o', color='navy', markersize=12, linewidth=3)
    title = 'System Cost vs. Asymmetric Penalty (rho)'
    ylabel = 'Resulting System Cost ($)'
    if log_scale:
        ax.set_yscale('log')
        title += ' (Log Scale)'
        ylabel += ' [Log Scale]'
    ax.set_title(title, fontdict=TITLE_FONT)
    ax.set_xlabel('Rho value (Weight on Over-prediction)', fontdict=LABEL_FONT)
    ax.set_ylabel(ylabel, fontdict=LABEL_FONT)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: f'${format(int(x), ",")}'))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

# --- Calling the plotting functions ---
if not df_composite_loss.empty:
    plot_composite_loss_vs_rmse(df_composite_loss, 'plot1_composite_loss.png')

if not df_all_models.empty:
    plot_performance_scatter_with_inset(df_all_models, 'plot2_performance_scatter.png')
    plot_economic_ranking(df_all_models, 'plot3_cost_ranking_linear.png', log_scale=False)
    plot_economic_ranking(df_all_models, 'plot3_cost_ranking_log.png', log_scale=True)
    plot_forecast_bias(df_all_models, 'plot4_bias_linear.png', symlog_scale=False)
    plot_forecast_bias(df_all_models, 'plot4_bias_symlog.png', symlog_scale=True)

if not df_asymmetric.empty:
    plot_rho_sensitivity(df_asymmetric, 'plot5_rho_sensitivity_linear.png', log_scale=False)
    plot_rho_sensitivity(df_asymmetric, 'plot5_rho_sensitivity_log.png', log_scale=True)

print("\nScript finished successfully.")
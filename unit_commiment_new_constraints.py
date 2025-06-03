import pyomo.environ as pyo

# Model Data 
# Number of time periods
T = 24 

# Generator Data
generators_data = [
    {
        "id": "G1", "Pmin": 50, "Pmax": 200, "Cost_Startup": 500, "Cost_NoLoad": 20, "Cost_Gen": 10,
        "MinUpTime": 3, "MinDownTime": 2, "Initial_Status": 0, "Initial_Time_On": 0, "Initial_Time_Off": 5,
        "RampUpRate": 60, "RampDownRate": 60, "Initial_Generation": 0 # MW/hr, MW
    },
    {
        "id": "G2", "Pmin": 80, "Pmax": 300, "Cost_Startup": 800, "Cost_NoLoad": 30, "Cost_Gen": 15,
        "MinUpTime": 4, "MinDownTime": 3, "Initial_Status": 1, "Initial_Time_On": 10, "Initial_Time_Off": 0,
        "RampUpRate": 100, "RampDownRate": 100, "Initial_Generation": 150 # MW/hr, MW (assuming it was generating something)
    },
    {
        "id": "G3", "Pmin": 20, "Pmax": 100, "Cost_Startup": 200, "Cost_NoLoad": 10, "Cost_Gen": 25, # Peaker
        "MinUpTime": 1, "MinDownTime": 1, "Initial_Status": 0, "Initial_Time_On": 0, "Initial_Time_Off": 2,
        "RampUpRate": 100, "RampDownRate": 100, "Initial_Generation": 0 # MW/hr, MW (can ramp its full capacity in one go)
    }
]

# System Demand (MW) for T periods
demand_data = [
    100, 110, 105, 100, 120, 150, 180, 220, 250, 260, 270, 280,
    275, 260, 240, 220, 200, 180, 160, 150, 140, 130, 120, 110
]
if len(demand_data) != T:
    raise ValueError(f"Demand data length ({len(demand_data)}) must match T ({T})")

# Pyomo Model
model = pyo.ConcreteModel(name="UnitCommitment_Ramp")

# Sets 
model.I_set = pyo.Set(initialize=[gen['id'] for gen in generators_data])
model.T_set = pyo.RangeSet(1, T) # 1-based indexing for time periods

# Parameters
model.Pmin = pyo.Param(model.I_set, initialize={gen['id']: gen['Pmin'] for gen in generators_data})
model.Pmax = pyo.Param(model.I_set, initialize={gen['id']: gen['Pmax'] for gen in generators_data})
model.Cost_Startup = pyo.Param(model.I_set, initialize={gen['id']: gen['Cost_Startup'] for gen in generators_data})
model.Cost_NoLoad = pyo.Param(model.I_set, initialize={gen['id']: gen['Cost_NoLoad'] for gen in generators_data})
model.Cost_Gen = pyo.Param(model.I_set, initialize={gen['id']: gen['Cost_Gen'] for gen in generators_data})
model.MinUpTime = pyo.Param(model.I_set, initialize={gen['id']: gen['MinUpTime'] for gen in generators_data})
model.MinDownTime = pyo.Param(model.I_set, initialize={gen['id']: gen['MinDownTime'] for gen in generators_data})
model.Initial_Status = pyo.Param(model.I_set, initialize={gen['id']: gen['Initial_Status'] for gen in generators_data})
model.Initial_Time_On = pyo.Param(model.I_set, initialize={gen['id']: gen['Initial_Time_On'] for gen in generators_data})
model.Initial_Time_Off = pyo.Param(model.I_set, initialize={gen['id']: gen['Initial_Time_Off'] for gen in generators_data})
model.Demand = pyo.Param(model.T_set, initialize={t: demand_data[t-1] for t in model.T_set})

# NEW: Ramp Rate Parameters
model.RampUpRate = pyo.Param(model.I_set, initialize={gen['id']: gen['RampUpRate'] for gen in generators_data})
model.RampDownRate = pyo.Param(model.I_set, initialize={gen['id']: gen['RampDownRate'] for gen in generators_data})
model.Initial_Generation = pyo.Param(model.I_set, initialize={gen['id']: gen['Initial_Generation'] for gen in generators_data})

# Decision Variables
model.z = pyo.Var(model.I_set, model.T_set, domain=pyo.Binary) # On/Off
model.y = pyo.Var(model.I_set, model.T_set, domain=pyo.Binary) # Startup
model.x = pyo.Var(model.I_set, model.T_set, domain=pyo.Binary) # Shutdown
model.g = pyo.Var(model.I_set, model.T_set, domain=pyo.NonNegativeReals) # Generation

# Objective Function : linear
def total_cost_rule(m):
    startup_c = sum(m.Cost_Startup[i] * m.y[i,t] for i in m.I_set for t in m.T_set)
    noload_c = sum(m.Cost_NoLoad[i] * m.z[i,t] for i in m.I_set for t in m.T_set)
    gen_c = sum(m.Cost_Gen[i] * m.g[i,t] for i in m.I_set for t in m.T_set)
    return startup_c + noload_c + gen_c
model.TotalCost = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

# Constraints

# 1. Generator ON/OFF Logic (Status, Startup, Shutdown)
model.StatusLogic = pyo.ConstraintList()
for i in model.I_set:
    for t in model.T_set:
        if t == 1:
            model.StatusLogic.add(model.z[i,t] - model.Initial_Status[i] == model.y[i,t] - model.x[i,t])
        else:
            model.StatusLogic.add(model.z[i,t] - model.z[i,t-1] == model.y[i,t] - model.x[i,t])
        model.StatusLogic.add(model.y[i,t] + model.x[i,t] <= 1)

# 2. Minimum Up Time
model.MinUpTimeConstraint = pyo.ConstraintList()
for i in model.I_set:
    if model.Initial_Status[i] == 1 and model.Initial_Time_On[i] < model.MinUpTime[i]:
        required_on_periods = model.MinUpTime[i] - model.Initial_Time_On[i]
        for t_init in range(1, min(T, required_on_periods) + 1):
            model.MinUpTimeConstraint.add(model.z[i, t_init] == 1)
            model.MinUpTimeConstraint.add(model.y[i, t_init] == 0)
            model.MinUpTimeConstraint.add(model.x[i, t_init] == 0)
    for t in model.T_set:
        for k in range(model.MinUpTime[i]):
            if t + k <= T:
                model.MinUpTimeConstraint.add(model.z[i, t + k] >= model.y[i, t])

# 3. Minimum Down Time
model.MinDownTimeConstraint = pyo.ConstraintList()
for i in model.I_set:
    if model.Initial_Status[i] == 0 and model.Initial_Time_Off[i] < model.MinDownTime[i]:
        required_off_periods = model.MinDownTime[i] - model.Initial_Time_Off[i]
        for t_init in range(1, min(T, required_off_periods) + 1):
            model.MinDownTimeConstraint.add(model.z[i, t_init] == 0)
            model.MinDownTimeConstraint.add(model.y[i, t_init] == 0)
            model.MinDownTimeConstraint.add(model.x[i, t_init] == 0)
    for t in model.T_set:
        for k in range(model.MinDownTime[i]):
            if t + k <= T:
                model.MinDownTimeConstraint.add(model.z[i, t + k] <= 1 - model.x[i, t])

# 4. Capacity / Generation Limits (Equation Set 5)
model.GenerationLimits = pyo.ConstraintList()
for i in model.I_set:
    for t in model.T_set:
        model.GenerationLimits.add(model.g[i,t] >= model.Pmin[i] * model.z[i,t])
        model.GenerationLimits.add(model.g[i,t] <= model.Pmax[i] * model.z[i,t])

# 5. Demand Balance (Equation Set 6)
model.DemandBalance = pyo.ConstraintList()
for t in model.T_set:
    model.DemandBalance.add(sum(model.g[i,t] for i in model.I_set) >= model.Demand[t])
# my notes
# 6. Ramp Constraints (Equation Set 7)
# Ramp limits are typically applied when the unit is ON in both t and t-1.
# More advanced formulations handle ramping up from zero (startup ramp) and ramping down to zero (shutdown ramp)
# common formulation:
# g[i,t] - g[i,t-1] <= RampUpRate[i]
# g[i,t-1] - g[i,t] <= RampDownRate[i]
# These apply when z[i,t]=1 and z[i,t-1]=1.
# If unit starts up at t (y[i,t]=1), g[i,t] <= Pmin[i] + RampUpRate_Startup[i] (often just Pmin or a special startup ramp)
# If unit shuts down at t (x[i,t]=1), g[i,t-1] <= Pmin[i] + RampDownRate_Shutdown[i] (often just Pmin or a special shutdown ramp)

# For simplicity and matching the paper's Eq (7) directly (git - gi(t-1)),
# which doesn't explicitly show dependence on z_it:
# I will apply them generally and let the generation limits g_it = 0 if z_it = 0 handle off states.
# However, a more robust formulation links ramp rates to online status.
#common formulation:
# g[i,t] - g[i,t-1] <= SU[i] * y[i,t] + RU[i] * z[i,t-1]  (SU is startup ramp, RU is normal ramp)
# g[i,t-1] - g[i,t] <= SD[i] * x[i,t] + RD[i] * z[i,t]    (SD is shutdown ramp, RD is normal ramp)
# implement the simpler version consistent with Eq. (7) structure,
# but be aware of these more detailed ramp considerations.

model.RampConstraints = pyo.ConstraintList()
for i in model.I_set:
    for t in model.T_set:
        if t == 1: # Ramp from initial generation
            # Ramp up from initial generation if unit is ON at t=1
            # model.RampConstraints.add(model.g[i,t] - model.Initial_Generation[i] <= model.RampUpRate[i] * model.z[i,t]) # More precise
            # model.RampConstraints.add(model.Initial_Generation[i] - model.g[i,t] <= model.RampDownRate[i] * model.z[i,t]) # More precise
            # Simplified based on paper's generic form, assumes g=0 if off.
            model.RampConstraints.add(model.g[i,t] - model.Initial_Generation[i] <= model.RampUpRate[i])
            model.RampConstraints.add(model.Initial_Generation[i] - model.g[i,t] <= model.RampDownRate[i])
        else: # Ramp from previous period's generation
            # model.RampConstraints.add(model.g[i,t] - model.g[i,t-1] <= model.RampUpRate[i] * model.z[i,t]) # More precise
            # model.RampConstraints.add(model.g[i,t-1] - model.g[i,t] <= model.RampDownRate[i] * model.z[i,t]) # More precise
            # Simplified based on paper's generic form
            model.RampConstraints.add(model.g[i,t] - model.g[i,t-1] <= model.RampUpRate[i])
            model.RampConstraints.add(model.g[i,t-1] - model.g[i,t] <= model.RampDownRate[i])


# Solve the Model using Gurobi
solver = pyo.SolverFactory('gurobi')
# solver.options['MIPGap'] = 0.01
# solver.options['TimeLimit'] = 300
results = solver.solve(model, tee=True)

# Display Results
if (results.solver.status == pyo.SolverStatus.ok) and \
   (results.solver.termination_condition == pyo.TerminationCondition.optimal):
    print("\n--- Optimal Solution Found (using Gurobi) ---")
    print(f"Total Cost: ${model.TotalCost():.2f}")

    print("\nUnit Status (z[i,t]):")
    print("Gen | " + " | ".join(f"T{t:<2}" for t in model.T_set))
    print("----|" + "-------" * T)
    for i in model.I_set:
        print(f"{i:<3} | " + " | ".join(f"{pyo.value(model.z[i,t]):<2.0f}" for t in model.T_set))

    print("\nUnit Startup (y[i,t]):")
    print("Gen | " + " | ".join(f"T{t:<2}" for t in model.T_set))
    print("----|" + "-------" * T)
    for i in model.I_set:
        print(f"{i:<3} | " + " | ".join(f"{pyo.value(model.y[i,t]):<2.0f}" for t in model.T_set))

    print("\nUnit Shutdown (x[i,t]):")
    print("Gen | " + " | ".join(f"T{t:<2}" for t in model.T_set))
    print("----|" + "-------" * T)
    for i in model.I_set:
        print(f"{i:<3} | " + " | ".join(f"{pyo.value(model.x[i,t]):<2.0f}" for t in model.T_set))

    print("\nGeneration Levels (g[i,t]) (MW):")
    print("Gen | " + " | ".join(f"  T{t:<4}" for t in model.T_set))
    print("----|" + "----------" * T)
    for i in model.I_set:
        print(f"{i:<3} | " + " | ".join(f"{pyo.value(model.g[i,t]):<6.1f}" for t in model.T_set))

    print("\nTotal Generation vs Demand (MW):")
    print("Time| Tot Gen | Demand  | Slack")
    print("----|---------|---------|---------")
    for t in model.T_set:
        total_gen_t = sum(pyo.value(model.g[i,t]) for i in model.I_set)
        demand_t = model.Demand[t]
        print(f"T{t:<2} | {total_gen_t:>7.1f} | {demand_t:>7.1f} | {(total_gen_t - demand_t):>7.1f}")

elif results.solver.termination_condition == pyo.TerminationCondition.infeasible:
    print("\n--- Model is Infeasible (using Gurobi) ---")
    print("Consider checking constraints, especially MinUp/MinDown initial conditions, demand levels, or RAMP constraints.")
    # model.write("infeasible_ramp_model.lp", io_options={'symbolic_solver_labels': True})
    # print("LP file 'infeasible_ramp_model.lp' written for debugging.")
else:
    print("\n--- Solver Status (Gurobi) ---")
    print("Status:", results.solver.status)
    print("Termination Condition:", results.solver.termination_condition)
    if hasattr(results.solver, 'message') and results.solver.message:
        print("Solver Message:", results.solver.message)

# model.pprint()
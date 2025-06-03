import pyomo.environ as pyo

# Model Configuration & Data 
T = 4 # I reduced for quick testing
BaseMVA = 100.0 # System Base MVA for per-unit calculations

# Network Data , I test just for 3 bus
nodes_data = {
    "Bus1": {"is_reference": True},
    "Bus2": {},
    "Bus3": {},
}

lines_data = {
    ("Bus1", "Bus2"): {"reactance_pu": 0.1, "capacity_mw": 150}, # X (p.u.), Capacity (MW)
    ("Bus1", "Bus3"): {"reactance_pu": 0.08, "capacity_mw": 100},
    ("Bus2", "Bus3"): {"reactance_pu": 0.05, "capacity_mw": 120},
}

# Generator Data 
# Added 'bus' attribute and quadratic cost coefficients
generators_data = [
    {
        "id": "G1", "bus": "Bus1", "Pmin": 50, "Pmax": 200, "Cost_Startup": 500, "Cost_NoLoad": 20,
        "Cost_Gen_Linear": 10, "Cost_Gen_Quadratic": 0.02,
        "MinUpTime": 3, "MinDownTime": 2, "Initial_Status": 0, "Initial_Time_On": 0, "Initial_Time_Off": T+1, # ensure it can start
        "RampUpRate": 200, "RampDownRate": 200, "Initial_Generation": 0 # Increased ramp for small T
    },
    {
        "id": "G2", "bus": "Bus2", "Pmin": 80, "Pmax": 300, "Cost_Startup": 800, "Cost_NoLoad": 30,
        "Cost_Gen_Linear": 15, "Cost_Gen_Quadratic": 0.01,
        "MinUpTime": 2, "MinDownTime": 2, "Initial_Status": 1, "Initial_Time_On": T+1, "Initial_Time_Off": 0, # ensure it can run
        "RampUpRate": 300, "RampDownRate": 300, "Initial_Generation": 150
    },
    {
        "id": "G3", "bus": "Bus3", "Pmin": 20, "Pmax": 100, "Cost_Startup": 200, "Cost_NoLoad": 10,
        "Cost_Gen_Linear": 25, "Cost_Gen_Quadratic": 0.05,
        "MinUpTime": 1, "MinDownTime": 1, "Initial_Status": 0, "Initial_Time_On": 0, "Initial_Time_Off": T+1,
        "RampUpRate": 100, "RampDownRate": 100, "Initial_Generation": 0
    }
]

# Demand Data (Nodal) 
# Original total system demand for T periods
original_total_demand = [100, 110, 105, 100, 120, 150, 180, 220, 250, 260, 270, 280,
                         275, 260, 240, 220, 200, 180, 160, 150, 140, 130, 120, 110]
# Slice for the current T
current_total_demand = original_total_demand[:T]

nodal_demand_data = { # nodal_demand_data[bus_id][time_period_index]
    "Bus1": [d * 0.2 for d in current_total_demand],
    "Bus2": [d * 0.5 for d in current_total_demand],
    "Bus3": [d * 0.3 for d in current_total_demand],
}
if any(len(nodal_demand_data[b]) != T for b in nodal_demand_data):
    raise ValueError("Nodal demand data length must match T")

# Pyomo Model 
model = pyo.ConcreteModel(name="NetworkUnitCommitment")

# Sets 
model.I_set = pyo.Set(initialize=[gen['id'] for gen in generators_data])  # Generators
model.T_set = pyo.RangeSet(1, T)                                          # Time periods
model.B_nodes = pyo.Set(initialize=nodes_data.keys())                     # Buses/Nodes
model.L_lines = pyo.Set(initialize=lines_data.keys(), dimen=2)            # Transmission Lines

# Parameters 
# Generator Parameters
model.Pmin = pyo.Param(model.I_set, initialize={gen['id']: gen['Pmin'] for gen in generators_data})
model.Pmax = pyo.Param(model.I_set, initialize={gen['id']: gen['Pmax'] for gen in generators_data})
model.Cost_Startup = pyo.Param(model.I_set, initialize={gen['id']: gen['Cost_Startup'] for gen in generators_data})
model.Cost_NoLoad = pyo.Param(model.I_set, initialize={gen['id']: gen['Cost_NoLoad'] for gen in generators_data})
model.Cost_Gen_Linear = pyo.Param(model.I_set, initialize={gen['id']: gen['Cost_Gen_Linear'] for gen in generators_data})
model.Cost_Gen_Quadratic = pyo.Param(model.I_set, initialize={gen['id']: gen['Cost_Gen_Quadratic'] for gen in generators_data})
model.MinUpTime = pyo.Param(model.I_set, initialize={gen['id']: gen['MinUpTime'] for gen in generators_data})
model.MinDownTime = pyo.Param(model.I_set, initialize={gen['id']: gen['MinDownTime'] for gen in generators_data})
model.Initial_Status = pyo.Param(model.I_set, initialize={gen['id']: gen['Initial_Status'] for gen in generators_data})
model.Initial_Time_On = pyo.Param(model.I_set, initialize={gen['id']: gen['Initial_Time_On'] for gen in generators_data})
model.Initial_Time_Off = pyo.Param(model.I_set, initialize={gen['id']: gen['Initial_Time_Off'] for gen in generators_data})
model.RampUpRate = pyo.Param(model.I_set, initialize={gen['id']: gen['RampUpRate'] for gen in generators_data})
model.RampDownRate = pyo.Param(model.I_set, initialize={gen['id']: gen['RampDownRate'] for gen in generators_data})
model.Initial_Generation = pyo.Param(model.I_set, initialize={gen['id']: gen['Initial_Generation'] for gen in generators_data})
model.GenBus = pyo.Param(model.I_set, initialize={gen['id']: gen['bus'] for gen in generators_data})

# Network Parameters
model.BaseMVA = pyo.Param(initialize=BaseMVA)
model.LineFrom = pyo.Param(model.L_lines, initialize={line: line[0] for line in model.L_lines})
model.LineTo = pyo.Param(model.L_lines, initialize={line: line[1] for line in model.L_lines})
model.LineReactance = pyo.Param(model.L_lines, initialize={line: data['reactance_pu'] for line, data in lines_data.items()})
model.LineCapacity = pyo.Param(model.L_lines, initialize={line: data['capacity_mw'] for line, data in lines_data.items()})
model.NodalDemand = pyo.Param(model.B_nodes, model.T_set,
                              initialize={(b, t_idx): nodal_demand_data[b][t_idx-1]
                                          for b in model.B_nodes for t_idx in model.T_set})
reference_bus_name = next(b for b, data in nodes_data.items() if data.get("is_reference", False))
model.ReferenceBus = pyo.Param(initialize=reference_bus_name, within=model.B_nodes)

# Helper Sets
model.GeneratorsAtBus = pyo.Set(model.B_nodes, initialize=lambda m, b: [i for i in m.I_set if m.GenBus[i] == b])
model.LinesFromBus = pyo.Set(model.B_nodes, initialize=lambda m, b: [l for l in m.L_lines if m.LineFrom[l] == b])
model.LinesToBus = pyo.Set(model.B_nodes, initialize=lambda m, b: [l for l in m.L_lines if m.LineTo[l] == b])

# Decision Variables
model.z = pyo.Var(model.I_set, model.T_set, domain=pyo.Binary, doc="On/Off status")
model.y = pyo.Var(model.I_set, model.T_set, domain=pyo.Binary, doc="Startup decision")
model.x = pyo.Var(model.I_set, model.T_set, domain=pyo.Binary, doc="Shutdown decision")
model.g = pyo.Var(model.I_set, model.T_set, domain=pyo.NonNegativeReals, doc="Generation level (MW)")

model.theta = pyo.Var(model.B_nodes, model.T_set, domain=pyo.Reals, doc="Bus voltage angle (radians)")
model.pline = pyo.Var(model.L_lines, model.T_set, domain=pyo.Reals, doc="Power flow on lines (MW)")

#  Objective Function 
def total_cost_rule_quadratic(m):
    startup_c = sum(m.Cost_Startup[i] * m.y[i,t] for i in m.I_set for t in m.T_set)
    noload_c = sum(m.Cost_NoLoad[i] * m.z[i,t] for i in m.I_set for t in m.T_set)
    gen_c = sum(
        m.Cost_Gen_Linear[i] * m.g[i,t] +
        m.Cost_Gen_Quadratic[i] * (m.g[i,t]**2)
        for i in m.I_set for t in m.T_set
    )
    return startup_c + noload_c + gen_c
model.TotalCost = pyo.Objective(rule=total_cost_rule_quadratic, sense=pyo.minimize)

# Constraints 

# 1. Generator ON/OFF Logic
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
            model.MinUpTimeConstraint.add(expr=model.z[i, t_init] == 1)
            model.MinUpTimeConstraint.add(expr=model.y[i, t_init] == 0) # Cannot startup if forced on
            model.MinUpTimeConstraint.add(expr=model.x[i, t_init] == 0) # Cannot shutdown if forced on
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
            model.MinDownTimeConstraint.add(expr=model.z[i, t_init] == 0)
            model.MinDownTimeConstraint.add(expr=model.y[i, t_init] == 0) # Cannot startup if forced off
            model.MinDownTimeConstraint.add(expr=model.x[i, t_init] == 0) # Cannot shutdown if forced off
    for t in model.T_set:
        for k in range(model.MinDownTime[i]):
            if t + k <= T:
                model.MinDownTimeConstraint.add(model.z[i, t + k] <= 1 - model.x[i, t])

# 4. Capacity / Generation Limits
model.GenerationLimits = pyo.ConstraintList()
for i in model.I_set:
    for t in model.T_set:
        model.GenerationLimits.add(model.g[i,t] >= model.Pmin[i] * model.z[i,t])
        model.GenerationLimits.add(model.g[i,t] <= model.Pmax[i] * model.z[i,t])

# 5. Ramp Constraints
model.RampConstraints = pyo.ConstraintList()
for i in model.I_set:
    for t in model.T_set:
        if t == 1:
            model.RampConstraints.add(model.g[i,t] - model.Initial_Generation[i] <= model.RampUpRate[i])
            model.RampConstraints.add(model.Initial_Generation[i] - model.g[i,t] <= model.RampDownRate[i])
        else:
            model.RampConstraints.add(model.g[i,t] - model.g[i,t-1] <= model.RampUpRate[i])
            model.RampConstraints.add(model.g[i,t-1] - model.g[i,t] <= model.RampDownRate[i])

# Network Constraints 

# 6. Nodal Power Balance
model.NodalPowerBalance = pyo.ConstraintList()
for t in model.T_set:
    for b_node in model.B_nodes:
        generation_at_bus = sum(model.g[i,t] for i in model.GeneratorsAtBus[b_node] if i in model.I_set) # check if i in model.I_set is important if GeneratorsAtBus can be empty
        demand_at_bus = model.NodalDemand[b_node,t]
        
        flow_out_of_bus = sum(model.pline[line,t] for line in model.LinesFromBus[b_node] if line in model.L_lines)
        flow_into_bus = sum(model.pline[line,t] for line in model.LinesToBus[b_node] if line in model.L_lines)

        model.NodalPowerBalance.add(
            generation_at_bus - demand_at_bus == flow_out_of_bus - flow_into_bus
        )

# 7. DC Power Flow Definition
model.DCPowerFlow = pyo.ConstraintList()
for t in model.T_set:
    for line in model.L_lines:
        from_bus = model.LineFrom[line]
        to_bus = model.LineTo[line]
        reactance = model.LineReactance[line]
        if reactance == 0:
            print(f"Warning: Zero reactance for line {line}. Skipping DC power flow constraint or add specific handling.")
            # Potentially add a constraint like model.pline[line,t] == 0 or handle based on B matrix if available
            continue # Or raise error
        
        # P_line = (BaseMVA / X_pu) * (theta_from - theta_to) if X is in p.u. and P_line in MW
        # If we use susceptance B = 1/X, then P_line = BaseMVA * B_pu * (theta_from - theta_to)
        # If all power values (Pmin, Pmax, Demand, LineCapacity, pline, g) are in MW,
        # and angles are in radians, and reactance is in p.u. on BaseMVA:
        model.DCPowerFlow.add(
            model.pline[line,t] == (model.BaseMVA / reactance) * \
                                     (model.theta[from_bus,t] - model.theta[to_bus,t])
        )

# 8. Line Flow Limits
model.LineFlowLimits = pyo.ConstraintList()
for t in model.T_set:
    for line in model.L_lines:
        model.LineFlowLimits.add(model.pline[line,t] <= model.LineCapacity[line])
        model.LineFlowLimits.add(model.pline[line,t] >= -model.LineCapacity[line])

# 9. Reference Bus Angle
model.ReferenceAngle = pyo.ConstraintList()
for t in model.T_set:
    model.ReferenceAngle.add(model.theta[model.ReferenceBus, t] == 0.0)


# --- Solve the Model ---
# I used Gurobi and accessible by Pyomo

# For Gurobi: pyo.SolverFactory('gurobi')


try:
    solver = pyo.SolverFactory('gurobi')
    # solver.options['MIPGap'] = 0.01
    # solver.options['TimeLimit'] = 600 # seconds
except Exception as e:
    print(f"Could not load Gurobi: {e}. Trying SCIP.")
    try:
        solver = pyo.SolverFactory('scip')
        # solver.options['limits/gap'] = 0.01
        # solver.options['limits/time'] = 600
    except Exception as e2:
        print(f"Could not load SCIP: {e2}. Please install a suitable MIQP solver.")
        exit()

results = solver.solve(model, tee=True)

# Display Results
if (results.solver.status == pyo.SolverStatus.ok) and \
   (results.solver.termination_condition == pyo.TerminationCondition.optimal or \
    results.solver.termination_condition == pyo.TerminationCondition.feasible): # SCIP might return feasible for MIQP within gap
    print("\n--- Solution Found ---")
    print(f"Total Cost: ${model.TotalCost():.2f}")

    print("\nUnit Status (z[i,t]):")
    header = "Gen | " + " | ".join(f"T{t_idx:<2}" for t_idx in model.T_set)
    print(header)
    print("----|" + "-------" * T)
    for i in model.I_set:
        row = f"{i:<3} | " + " | ".join(f"{pyo.value(model.z[i,t_idx]):<2.0f}" for t_idx in model.T_set)
        print(row)

    print("\nGeneration Levels (g[i,t]) (MW):")
    header = "Gen | " + " | ".join(f"  T{t_idx:<4}" for t_idx in model.T_set)
    print(header)
    print("----|" + "----------" * T)
    for i in model.I_set:
        row = f"{i:<3} | " + " | ".join(f"{pyo.value(model.g[i,t_idx]):<6.1f}" for t_idx in model.T_set)
        print(row)

    print("\nBus Voltage Angles (theta[b,t]) (radians):")
    header = "Bus | " + " | ".join(f"  T{t_idx:<5}" for t_idx in model.T_set)
    print(header)
    print("----|" + "-----------" * T)
    for b_node in model.B_nodes:
        row = f"{b_node:<3} | " + " | ".join(f"{pyo.value(model.theta[b_node,t_idx]):<7.3f}" for t_idx in model.T_set)
        print(row)

    print("\nLine Power Flows (pline[l,t]) (MW):")
    header = "Line        | " + " | ".join(f"  T{t_idx:<6}" for t_idx in model.T_set)
    print(header)
    print("------------|" + "------------" * T)
    for line in model.L_lines:
        line_str = f"{model.LineFrom[line]}-{model.LineTo[line]}"
        row = f"{line_str:<11} | " + " | ".join(f"{pyo.value(model.pline[line,t_idx]):<8.1f}" for t_idx in model.T_set)
        print(row)
    
    print("\nNodal Power Balance Check (Gen - Demand - NetFlowOut == 0):")
    header = "Bus | Time | Gen   | Demand| NetFlowOut | Balance"
    print(header)
    for t_idx in model.T_set:
        for b_node in model.B_nodes:
            gen_val = sum(pyo.value(model.g[i,t_idx]) for i in model.GeneratorsAtBus[b_node] if i in model.I_set)
            dem_val = model.NodalDemand[b_node,t_idx]
            flow_out = sum(pyo.value(model.pline[line,t_idx]) for line in model.LinesFromBus[b_node] if line in model.L_lines)
            flow_in = sum(pyo.value(model.pline[line,t_idx]) for line in model.LinesToBus[b_node] if line in model.L_lines)
            net_flow_out = flow_out - flow_in
            balance = gen_val - dem_val - net_flow_out
            print(f"{b_node:<3} | T{t_idx:<2}  | {gen_val:<5.1f} | {dem_val:<5.1f} | {net_flow_out:<10.1f} | {balance:<7.2f}")


elif results.solver.termination_condition == pyo.TerminationCondition.infeasible:
    print("\n--- Model is Infeasible ---")
    print("Check constraints, especially initial conditions, demand levels, capacities, or ramp rates.")
    #just for my notes
    # For debugging infeasibility:
    # model.compute_unbounded_rays() # If unbounded
    # model.compute_slack_variables() # If infeasible
    # from pyomo.util.infeasible import log_infeasible_constraints
    # log_infeasible_constraints(model, log_expression=True, log_variables=True)
    # model.write("infeasible_ncuc_model.lp", io_options={'symbolic_solver_labels': True})
    # print("LP file 'infeasible_ncuc_model.lp' written for debugging.")

else:
    print("\n--- Solver Status ---")
    print("Status:", results.solver.status)
    print("Termination Condition:", results.solver.termination_condition)
    if hasattr(results.solver, 'message') and results.solver.message:
        print("Solver Message:", results.solver.message)

# model.pprint() # this is to print the full model structure like previous
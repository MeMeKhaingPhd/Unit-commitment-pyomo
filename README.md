


# Network-Constrained Unit Commitment (NCUC) with Quadratic Costs in Pyomo

## Project Overview

This project implements a Network-Constrained Unit Commitment (NCUC) optimization model using Python and the Pyomo modeling language. The model aims to determine the optimal generation schedule for a set of power generating units over a defined time horizon, considering both generator operational constraints and power transmission network limitations. The objective is to minimize the total operational cost, which includes startup costs, no-load costs, and quadratic generation costs.

## Model Features

The Pyomo-based NCUC model incorporates the following key features:

1.  Objective Function:
    - Minimization of total system operational cost.
    - Cost components:
        *   **Startup Cost:** Incurred when a unit is turned on.
        *   **No-Load Cost:** Cost of keeping a unit synchronized to the grid, even at zero output.
        *   **Generation Cost:** Modeled with a quadratic cost curve (`a*g + b*g^2`), providing a more realistic representation of fuel consumption.

2.  **Generator Constraints:**
    *   **On/Off Logic:** Tracks the status (on/off, startup, shutdown) of each unit in each time period.
    *   **Minimum Up Time:** Once a unit is started, it must remain online for a minimum duration.
    *   **Minimum Down Time:** Once a unit is shut down, it must remain offline for a minimum duration.
    *   **Generation Limits:** Each online unit must operate between its minimum (Pmin) and maximum (Pmax) generation capacity.
    *   **Ramp Rate Limits:** Constraints on how quickly a unit can increase (ramp-up) or decrease (ramp-down) its generation output between consecutive time periods.
    *   **Initial Conditions:** Accounts for the initial status, time in that status, and generation level of each unit before the start of the planning horizon.

3.  **Network Constraints (DC Power Flow Model):**
    *   **Nodal Power Balance:** At each bus (node) in the network, for every time period, the total power generated at that bus minus the total demand at that bus must equal the net power flowing out of the bus onto transmission lines.
    *   **DC Power Flow Equations:** Line power flows are determined based on the difference in voltage angles between connected buses and the line's reactance. (This model uses an explicit formulation with bus voltage angle variables).
    *   **Line Flow Limits:** Each transmission line has a maximum power carrying capacity (thermal limit), which cannot be exceeded in either direction.
    *   **Reference Bus:** One bus is designated as the reference bus with a fixed voltage angle (typically 0 radians) to solve the power flow equations.
    *   **Nodal Demand:** System demand is specified on a per-bus, per-time-period basis.

4.  **Problem Formulation:**
    *   The model is formulated as a **Mixed-Integer Quadratic Program (MIQP)** due to the binary variables (on/off, startup/shutdown decisions) and the quadratic terms in the generation cost function.

## Implementation Details

*   **Language:** Python 3.x
*   **Modeling Tool:** Pyomo
*   **Solvers:** The model is designed to be solved using MIQP-capable solvers such as Gurobi
*   **Data Input:** Generator characteristics, network topology (buses, lines, reactances, capacities), and nodal demand profiles are defined within the Python script using dictionaries and lists. (For larger systems, this data could be read from external files like CSV or Excel).*

# How to Run

1.  **Prerequisites:**
    *   Python 3.x
    *   Pyomo: `pip install pyomo`
    *   A compatible MIQP solver Gurobi.
        *   For Gurobi: Ensure Gurobi is installed and its license is configured. The `gurobipy` Python package is also needed (`pip install gurobipy`).
       
2.  **Configure Data:**
    *   Modify the data sections at the beginning of the Python script (`ncuc_model.py` or similar) to define system:
        *   `T`: Number of time periods.
        *   `BaseMVA`: System base MVA.
        *   `nodes_data`: Bus information, including the reference bus.
        *   `lines_data`: Line connections, reactances (p.u.), and capacities (MW).
        *   `generators_data`: Generator parameters (Pmin, Pmax, costs, ramp rates, initial conditions, bus location).
        *   `nodal_demand_data`: Demand at each bus for each time period.

3.  **Execute the Script:**
   
    python name.py : this is example, my python name is unit_commmiment_MIQP_Network.py
   

4.  **Output:**
    *   The solver log will be printed to the console as I tested
    *   If a solution is found, the total minimized cost, unit status, generation levels, bus voltage angles, and line power flows will be displayed.
    *   A nodal power balance check is also performed and printed.

## Example System

The provided script includes a small, illustrative 3-bus system with 3 generators to demonstrate the model's functionality. To model a larger system (e.g., the IEEE 24-bus system), the data structures would need to be populated accordingly.

## Future Work / Potential Extensions

*   Implementation of AC Optimal Power Flow (ACOPF) for more accuracy (results in a MINLP).
*   Inclusion of reserve requirements and ancillary services.
*   Modeling of energy storage systems.
*   Consideration of generator ramp-up/down costs or shutdown costs.
*   Stochastic unit commitment to handle uncertainties (e.g., in renewable generation or demand).
*   Security constraints (N-1 contingency analysis).
*   Reading data from external files (e.g., MATPOWER case files, CSVs).

## Acknowledgements

 Aswin Sir suggestion however it may be different what you want please let me know.

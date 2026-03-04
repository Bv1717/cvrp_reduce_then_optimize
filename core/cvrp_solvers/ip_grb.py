# """ LP/MIP models and helper functions to solve TP and FCTP instances. """

##Changes
""" LP/MIP models and helper functions to solve CVRP instances. """
##End changes

import random

import gurobipy as gp
import networkx as nx
import numpy as np
from VRPSolverEasy.src import solver

# from core.utils.preprocessing import get_weight_matrix

##################################################
# LP and MIP Models
##################################################
 

# def tp(supplies, demands, costs):
#     """Generate Gurobi model for the TP.

#     Parameters
#     ----------
#     supplies: 1D np.array or list
#         A list of supplier capacities/supplies.
#     demands: 1D np.array or list
#         A list of customer demands.
#     costs: 2D np.array
#         Costs for sending one unit of flow form supplier i to customer j.

#     Returns
#     -------
#     m: gp.Model
#         Gurobi model of TP instance.
#     x: dict
#         Dictionary of flow variables, keyed by (supplier, customer).

#     """

#     m = gp.Model()

#     # The flow variables represent the absolute flow from a certain supplier to a certain customer
#     x = m.addVars(*costs.shape, obj=costs, vtype=gp.GRB.CONTINUOUS, name="x")

#     # For each customer, ensure that the total demand is fulfilled
#     for j, demand in enumerate(demands):
#         m.addConstr(
#             gp.quicksum(x[i, j] for i in range(len(supplies)) if (i, j) in x) == demand,
#             name=f"B{j}",
#         )

#     # For each supplier, ensure that the total capacity is not exceeded (unbalanced) or
#     # fully distributed (balanced)
#     balanced = sum(supplies) == sum(demands)
#     if balanced:
#         for i, supply in enumerate(supplies):
#             m.addConstr(
#                 gp.quicksum(x[i, j] for j in range(len(demands)) if (i, j) in x)
#                 == supply,
#                 name=f"A{i}",
#             )
#     else:
#         for i, supply in enumerate(supplies):
#             m.addConstr(
#                 gp.quicksum(x[i, j] for j in range(len(demands)) if (i, j) in x)
#                 <= supply,
#                 name=f"A{i}",
#             )

#     return m, x


# def fctp(supplies, demands, var_costs, fix_costs, relax=False):
#     """Generate Gurobi model for the FCTP.

#     The FCTP extends the TP by including fixed costs on edges.

#     Parameters
#     ----------
#     supplies: 1D np.array or list
#         A list of supplier capacities/supplies.
#     demands: 1D np.array or list
#         A list of customer demands.
#     var_costs: 2D np.array
#         Variables costs for sending one unit of flow form supplier i to customer j.
#     fix_costs: 2D np.array
#         Fixed costs for sending flow form supplier i to customer j.
#     relax: bool, optional
#         Indicate whether the fixed cost variables should be linearized. Default is False.

#     Returns
#     -------
#     m: gp.Model
#         Gurobi model of FCTP instance.
#     x: dict
#         Dictionary of flow variables, keyed by (supplier, customer).
#     y: dict
#         Dictionary of activation variables, keyed by (supplier, customer).

#     """

#     # Get TP model
#     m, x = tp(supplies, demands, var_costs)

#     # Add activation variables that indicate whether a certain connection is used
#     if relax:
#         y = m.addVars(
#             *fix_costs.shape, obj=fix_costs, vtype=gp.GRB.CONTINUOUS, name="y"
#         )
#     else:
#         y = m.addVars(*fix_costs.shape, obj=fix_costs, vtype=gp.GRB.BINARY, name="y")

#     # Set activation variable to one if the connection is used by some flow
#     for i, j in x.keys():
#         big_m = min(supplies[i], demands[j])
#         m.addConstr(x[i, j] <= big_m * y[i, j], name=f"G[{i},{j}]")

#     return m, x, y


# def capacitated_fctp(
#     supplies, demands, var_costs, fix_costs, edge_capacities, relax=False
# ):
#     """Generate Gurobi model for the FCTP with edge capacities.

#     Parameters
#     ----------
#     supplies: 1D np.array or list
#         A list of supplier capacities/supplies.
#     demands: 1D np.array or list
#         A list of customer demands.
#     var_costs: 2D np.array
#         Variables costs for sending one unit of flow form supplier i to customer j.
#     fix_costs: 2D np.array
#         Fixed costs for sending flow form supplier i to customer j.
#     edge_capacities: 2D np.array
#         Maximum flow on edge.
#     relax: bool, optional
#         Indicate whether the fixed cost variables should be linearized. Default is False.

#     Returns
#     -------
#     m: gp.Model
#         Gurobi model of FCTP instance.
#     x: dict
#         Dictionary of flow variables, keyed by (supplier, customer).
#     y: dict
#         Dictionary of activation variables, keyed by (supplier, customer).

#     """

#     # Get FCTP IP
#     m, x, y = fctp(supplies, demands, var_costs, fix_costs, relax)

#     # Add constraint to ensure that edge capacities are not exceeded
#     for i, j in x.keys():
#         m.addConstr(x[i, j] <= edge_capacities[i, j], name=f"C[{i},{j}]")

#     return m, x, y


# def fixed_step_fctp(
#     supplies, demands, var_costs, fix_costs, vehicle_capacities, relax=False
# ):
#     """Generate Gurobi model for the FCTP with vehicle capacities and fixed-step costs.

#     Parameters
#     ----------
#     supplies: 1D np.array or list
#         A list of supplier capacities/supplies.
#     demands: 1D np.array or list
#         A list of customer demands.
#     var_costs: 2D np.array
#         Variables costs for sending one unit of flow form supplier i to customer j.
#     fix_costs: 2D np.array
#         Fixed costs for sending flow form supplier i to customer j.
#     vehicle_capacities: 2D np.array
#         Maximum quantity per vehicle.
#     relax: bool, optional
#         Indicate whether the fixed cost variables should be linearized. Default is False.

#     Returns
#     -------
#     m: gp.Model
#         Gurobi model of FCTP instance.
#     x: dict
#         Dictionary of flow variables, keyed by (supplier, customer).
#     y: dict
#         Dictionary of activation variables, keyed by (supplier, customer).

#     """
#     # Get TP model
#     m, x = tp(supplies, demands, var_costs)

#     # Add vehicle counter variables
#     if relax:
#         y = m.addVars(
#             *fix_costs.shape, obj=fix_costs, vtype=gp.GRB.CONTINUOUS, name="y"
#         )
#     else:
#         y = m.addVars(*fix_costs.shape, obj=fix_costs, vtype=gp.GRB.INTEGER, name="y")

#     for i, j in x.keys():
#         m.addConstr(x[i, j] <= vehicle_capacities[i, j] * y[i, j], name=f"G[{i},{j}]")

#     return m, x, y


# def tp_subset_connections(supplies, demands, costs, connections):
#     """Generate Gurobi model for the TP with a reduced subset of connections.

#     Parameters
#     ----------
#     supplies: 1D np.array or list
#         A list of supplier capacities/supplies.
#     demands: 1D np.array or list
#         A list of customer demands.
#     costs: 2D np.array
#         Costs for sending one unit of flow form supplier i to customer j.
#     connections: 2D np.array
#         Binary adjacency matrix indicating whether connection should be included.

#     Returns
#     -------
#     m: gp.Model
#         Gurobi model of TP instance.
#     x: dict
#         Dictionary of flow variables, keyed by (supplier, customer).

#     """

#     m = gp.Model()

#     x = dict()
#     for i, _ in enumerate(supplies):
#         for j, _ in enumerate(demands):
#             if connections[i, j]:
#                 # The flow variables represent the absolute flow from a certain supplier to a certain customer
#                 x[i, j] = m.addVar(
#                     obj=costs[i, j], vtype=gp.GRB.CONTINUOUS, name=f"x[{i},{j}]"
#                 )

#     # For each customer, ensure that the total demand is fulfilled
#     for j, demand in enumerate(demands):
#         m.addConstr(
#             gp.quicksum(x[i, j] for i in range(len(supplies)) if (i, j) in x) == demand,
#             name=f"B{j}",
#         )

#     # For each supplier, ensure that the total capacity is not exceeded
#     balanced = sum(supplies) == sum(demands)
#     if balanced:
#         for i, supply in enumerate(supplies):
#             m.addConstr(
#                 gp.quicksum(x[i, j] for j in range(len(demands)) if (i, j) in x)
#                 == supply,
#                 name=f"A{i}",
#             )
#     else:
#         for i, supply in enumerate(supplies):
#             m.addConstr(
#                 gp.quicksum(x[i, j] for j in range(len(demands)) if (i, j) in x)
#                 <= supply,
#                 name=f"A{i}",
#             )

#     return m, x


# def fctp_subset_connections(
#     supplies, demands, var_costs, fix_costs, connections, relax=False
# ):
#     """Generate Gurobi model for the FCTP with a reduced subset of connections.

#     Parameters
#     ----------
#     supplies: 1D np.array or list
#         A list of supplier capacities/supplies.
#     demands: 1D np.array or list
#         A list of customer demands.
#     var_costs: 2D np.array
#         Variables costs for sending one unit of flow form supplier i to customer j.
#     fix_costs: 2D np.array
#         Fixed costs for sending flow form supplier i to customer j.
#     connections: 2D np.array
#         Binary adjacency matrix indicating whether connection should be included.
#     relax: bool, optional
#         Indicate whether the fixed cost variables should be linearized. Default is False.

#     Returns
#     -------
#     m: gp.Model
#         Gurobi model of FCTP instance.
#     x: dict
#         Dictionary of flow variables, keyed by (supplier, customer).
#     y: dict
#         Dictionary of activation variables, keyed by (supplier, customer).

#     """

#     m, x = tp_subset_connections(supplies, demands, var_costs, connections)

#     y = dict()
#     for i, _ in enumerate(supplies):
#         for j, _ in enumerate(demands):
#             if connections[i, j]:
#                 # The activation variables indicate whether a certain connection is used
#                 if relax:
#                     y[i, j] = m.addVar(
#                         obj=fix_costs[i, j], vtype=gp.GRB.CONTINUOUS, name=f"y[{i},{j}]"
#                     )
#                 else:
#                     y[i, j] = m.addVar(
#                         obj=fix_costs[i, j], vtype=gp.GRB.BINARY, name=f"y[{i},{j}]"
#                     )

#     # Set activation variable to one if the connection is used by some flow
#     for i, j in x.keys():
#         big_m = min(supplies[i], demands[j])
#         m.addConstr(x[i, j] <= big_m * y[i, j])

#     return m, x, y


# def capacitated_fctp_subset_connections(
#     supplies, demands, var_costs, fix_costs, edge_capacities, connections, relax=False
# ):
#     """Generate Gurobi model for the FCTP with edge capacities.

#     Parameters
#     ----------
#     supplies: 1D np.array or list
#         A list of supplier capacities/supplies.
#     demands: 1D np.array or list
#         A list of customer demands.
#     var_costs: 2D np.array
#         Variables costs for sending one unit of flow form supplier i to customer j.
#     fix_costs: 2D np.array
#         Fixed costs for sending flow form supplier i to customer j.
#     edge_capacities: 2D np.array
#         Maximum flow on edge.
#     connections: 2D np.array
#         Binary adjacency matrix indicating whether connection should be included.
#     relax: bool, optional
#         Indicate whether the fixed cost variables should be linearized. Default is False.

#     Returns
#     -------
#     m: gp.Model
#         Gurobi model of FCTP instance.
#     x: dict
#         Dictionary of flow variables, keyed by (supplier, customer).
#     y: dict
#         Dictionary of activation variables, keyed by (supplier, customer).

#     """

#     # Get FCTP IP
#     m, x, y = fctp_subset_connections(
#         supplies, demands, var_costs, fix_costs, connections, relax
#     )

#     # Add constraint to ensure that edge capacities are not exceeded
#     for i, j in x.keys():
#         m.addConstr(x[i, j] <= edge_capacities[i, j], name=f"C[{i},{j}]")

#     return m, x, y


# def fixed_step_fctp_subset_connections(
#     supplies,
#     demands,
#     var_costs,
#     fix_costs,
#     vehicle_capacities,
#     connections,
#     relax=False,
# ):
#     """Generate Gurobi model for the FCTP with a reduced subset of connections.

#     Parameters
#     ----------
#     supplies: 1D np.array or list
#         A list of supplier capacities/supplies.
#     demands: 1D np.array or list
#         A list of customer demands.
#     var_costs: 2D np.array
#         Variables costs for sending one unit of flow form supplier i to customer j.
#     fix_costs: 2D np.array
#         Fixed costs for sending flow form supplier i to customer j.
#     vehicle_capacities: 2D np.array
#         Maximum quantity per vehicle.
#     connections: 2D np.array
#         Binary adjacency matrix indicating whether connection should be included.
#     relax: bool, optional
#         Indicate whether the fixed cost variables should be linearized. Default is False.

#     Returns
#     -------
#     m: gp.Model
#         Gurobi model of FCTP instance.
#     x: dict
#         Dictionary of flow variables, keyed by (supplier, customer).
#     y: dict
#         Dictionary of activation variables, keyed by (supplier, customer).

#     """

#     m, x = tp_subset_connections(supplies, demands, var_costs, connections)

#     y = dict()
#     for i, _ in enumerate(supplies):
#         for j, _ in enumerate(demands):
#             if connections[i, j]:
#                 # The activation variables indicate whether a certain connection is used
#                 if relax:
#                     y[i, j] = m.addVar(
#                         obj=fix_costs[i, j], vtype=gp.GRB.CONTINUOUS, name=f"y[{i},{j}]"
#                     )
#                 else:
#                     y[i, j] = m.addVar(
#                         obj=fix_costs[i, j], vtype=gp.GRB.INTEGER, name=f"y[{i},{j}]"
#                     )

#     # Set activation variable to one if the connection is used by some flow
#     for i, j in x.keys():
#         m.addConstr(x[i, j] <= vehicle_capacities[i, j] * y[i, j])

#     return m, x, y


##################################################
# Helper functions
##################################################


def sol_vals(var_dict):
    """Translate solution into a dictionary.

    Parameters
    ----------
    var_dict: dict
        Dictionary containing model variables.

    Returns
    -------
    dict
        Dictionary containing variable values.

    """
    return {k: np.round(v.X) for k, v in var_dict.items()}


# def get_fctp_bfs(instance, edge_mask=None):
#     """Get basic feasible solution for (reduced) FCTP.

#     Step 1: Solve linear relaxation
#     Step 2: Create proper basic feasible solution

#     Parameters
#     ----------
#     instance: FCTP
#         FCTP instance to be solved.
#     edge_mask: 2D np.array, optional
#         Binary adjacency matrix indicating whether connection should be included.

#     Returns
#     -------
#     bfs: dict
#         Basic feasible solution {(i,j): flow}.

#     """
#     lp_costs = get_weight_matrix(
#         instance.var_costs,
#         instance.fix_costs,
#         instance.supply,
#         instance.demand,
#         only_var_costs=False,
#         inverse=False,
#     )
#     if edge_mask is None:
#         model, x = tp(instance.supply, instance.demand, lp_costs)
#     else:
#         model, x = tp_subset_connections(
#             instance.supply, instance.demand, lp_costs, edge_mask
#         )
#     model.setParam("OutputFlag", 0)
#     model.optimize()
#     bfs = get_basis(x, len(instance.supply), len(instance.demand))
#     return bfs


# def get_basis(x, m, n):
#     """Get proper basis by filling up solution with zero-edges.

#     Note: A BFS for the FCTP contains m + n - 1 edges that form a spanning tree.

#     Parameters
#     ----------
#     x: dict
#         Dictionary of flow variables, keyed by (supplier, customer).
#     m: int
#         Number of suppliers.
#     n: int
#         Number of customers.

#     Returns
#     -------
#     bfs: dict
#         Basic feasible solution {(i,j): flow}.

#     """
#     bfs = {}
#     nbv = []
#     for k, v in x.items():
#         if v.vbasis == 0:
#             bfs[k] = v.X
#         else:
#             nbv.append(k)

#     size = m + n - 1
#     if not len(bfs) == size:
#         bfs = fill_basis_with_nbv(bfs, nbv, m, n, max_tries=50)

#     assert len(bfs) == m + n - 1, "Basis does not have the correct size"
#     return bfs


# def fill_basis_with_nbv(bfs, nbv, m, n, max_tries=50):
#     """Get proper basis by filling up solution with zero-edges.

#     Note: A BFS for the FCTP contains m + n - 1 edges that form a spanning tree.

#     This functions sequentially adds zero-value edges to a solution until a proper basis is
#     obtained.

#     Parameters
#     ----------
#     bfs: dict
#         Potential basic feasible solution {(i,j): flow}.
#     nbv: list
#         List of non-basic variables.
#     m: int
#         Number of suppliers.
#     n: int
#         Number of customers.
#     max_tries: int, optional
#         Maximum number of retries with different sequence of NBVs if first try was unsuccessful.

#     Returns
#     -------
#     bfs: dict
#         Basic feasible solution {(i,j): flow}.

#     """

#     size = m + n - 1

#     # make sure that the bfs indeed forms a spanning tree
#     spanning_tree_graph = get_spanning_tree(bfs, m, n)
#     try:
#         nx.find_cycle(spanning_tree_graph)
#         raise Exception("Spanning tree contains cycles")
#     except:
#         pass

#     for i in range(max_tries):
#         # print(f"attempt {i}")
#         bfs_tmp = bfs.copy()
#         spanning_tree_graph_tmp = spanning_tree_graph.copy()

#         # add zero variables into the basis if they do not create a cycle until the target size has been reached
#         for k in nbv:
#             # tentatively add NBV to basis and check whether a cycle is formed
#             spanning_tree_copy = spanning_tree_graph_tmp.copy()
#             spanning_tree_copy.add_edge(f"S{k[0]}", f"D{k[1]}")
#             try:
#                 nx.find_cycle(spanning_tree_copy)
#                 continue
#             except:
#                 bfs_tmp[k] = 0
#                 spanning_tree_graph_tmp.add_edge(f"S{k[0]}", f"D{k[1]}")
#                 if len(bfs_tmp) == size:
#                     return bfs_tmp

#         # add NBVs in a different sequence
#         random.shuffle(nbv)
#     return bfs


# def get_spanning_tree(bfs, m, n):
#     """Get Spanning Tree Graph representing BFS.

#     Parameters
#     ----------
#     bfs: dict
#         Basic feasible solution {(i,j): flow}.
#     m: int
#         Number of suppliers.
#     n: int
#         Number of customers.

#     Returns
#     -------
#     graph: nx.Graph
#         Spanning Tree Graph.

#     """
#     graph = nx.Graph()
#     supply_nodes = [f"S{i}" for i in range(m)]
#     demand_nodes = [f"D{j}" for j in range(n)]
#     edges = [(f"S{i}", f"D{j}") for i, j in bfs.keys()]
#     graph.add_nodes_from(supply_nodes)
#     graph.add_nodes_from(demand_nodes)
#     graph.add_edges_from(edges)
#     return graph


##Changes


def cvrp(demands, arc_index, arc_costs, nb_vehicles, vehicle_capacity, sol_dict = None):
    """Generate Gurobi model for the CVRP.

    Parameters
    ----------
    demands: 1D np.array or list
        A list of customer demands, a null demand corresponds to the depot.
    arc_index : 2D np.array
        List of arcs with source and destination
    costs: 1D np.array
        Costs for using the ith arc.
    nb_vehicles: int
        Number of vehicles.
    vehicle_capacity : int or float
        Uniform capacity for the vehicles.
    relax : bool, optional
        Indicate whether the cost variables should be linearized. Default is False.


    Returns
    -------
    m: gp.Model
        Gurobi model of TP instance.
    x: dict
        Dictionary of arc activation variables, keyed by (node.id(i), node.id(j)).
    u: dict
        Dictionnary of used capacity after delivering city j, keyed by node.id(j).

    """



    m = gp.Model()

    arc_list = [(int(src), int(dst)) for src, dst in zip(arc_index[0], arc_index[1])]

    cost_dict = {(int(src), int(dst)): arc_costs[k]
             for k, (src, dst) in enumerate(zip(arc_index[0], arc_index[1]))}
    # print(cost_dict)

    x = m.addVars(arc_list, obj=cost_dict, vtype=gp.GRB.BINARY, name="x")

    if sol_dict is not None:
        for arc, value in sol_dict.items():
            if arc in x:
                x[arc].start = value

    # The activation variables tell whether an arc is used to travel from a place to another

    u = m.addVars(len(demands), vtype=gp.GRB.CONTINUOUS, name="u")

    # For each pair of customers, first MTZ inequality
    for i, demand_i in enumerate(demands):
        for j, demand_j in enumerate(demands):
            if i != j and demand_i > 0 and demand_j > 0 and (i, j) in arc_list:
                m.addConstr(
                    u[j] - u[i] >= demand_j - vehicle_capacity * (1 - x[i, j]),
                    name=f"MTZ1_{i}_{j}"
                )

    # For each customer, second MTZ inequality
    for j, demand_j in enumerate(demands):
        if(demand_j > 0):
            m.addConstr(u[j] >= demand_j, name=f"MTZ2_lb{j}")
            m.addConstr(u[j] <= vehicle_capacity, name=f"MTZ2_ub{j}")

    # Constraint for the number of vehicles leaving the depot
    for i, demand in enumerate(demands):
        if(demand == 0): 
            m.addConstr(
                gp.quicksum(x[i, j] for j in range(len(demands)) 
                            if (i, j) in x and i != j ) <= nb_vehicles,
                name=f"CV{i}",
            )
            break
    # Leaving customer constraint
    for i, demand in enumerate(demands):
        if(demand > 0):
            m.addConstr(
                gp.quicksum(x[i, j] for j in range(len(demands))
                            if (i, j) in x and i != j) == 1,
                name=f"LC{i}"
            )
    
    #Serving customer constraint
    for j, demand in enumerate(demands):
        if(demand > 0):
            m.addConstr(
                gp.quicksum(x[i, j] for i in range(len(demands))
                            if (i, j) in x and i != j) == 1,
                name=f"SC{j}"
            )

    return m, x, u

def cvrptw(
    demands,
    arc_index,
    arc_costs,
    nb_vehicles,
    vehicle_capacity,
    ready_time,
    due_time,
    service_time,
    sol_dict=None,
):
    """Generate Gurobi model for the CVRPTW (CVRP with time windows).

    Parameters
    ----------
    demands: 1D np.array or list
        Customer demands. A zero demand corresponds to the depot.
    arc_index : 2D np.array
        List of arcs with source and destination
    arc_costs: 1D np.array
        Cost / travel-time proxy for using the kth arc.
    nb_vehicles: int
        Number of vehicles.
    vehicle_capacity : int or float
        Uniform capacity for the vehicles.
    ready_time : 1D np.array
        Earliest service start time a_i for each node i.
    due_time : 1D np.array
        Latest service start time b_i for each node i.
    service_time : 1D np.array
        Service duration s_i for each node i.
    sol_dict : dict, optional
        Warm-start solution for x.

    Returns
    -------
    m: gp.Model
        Gurobi model.
    x: dict
        Arc activation variables x[i,j] in {0,1}.
    u: dict
        MTZ capacity variables.
    t: dict
        Service start time variables.
    """

    m = gp.Model()

    arc_list = [(int(src), int(dst)) for src, dst in zip(arc_index[0], arc_index[1])]

    cost_dict = {
        (int(src), int(dst)): arc_costs[k]
        for k, (src, dst) in enumerate(zip(arc_index[0], arc_index[1]))
    }

    # x[i,j] = 1 if arc (i,j) is used
    x = m.addVars(arc_list, obj=cost_dict, vtype=gp.GRB.BINARY, name="x")

    # Optional warm start
    if sol_dict is not None:
        for arc, value in sol_dict.items():
            if arc in x:
                x[arc].start = value

    # MTZ capacity variables (same as CVRP)
    u = m.addVars(len(demands), vtype=gp.GRB.CONTINUOUS, name="u")

    # CHANGE (TW): service start time variables
    t = m.addVars(len(demands), vtype=gp.GRB.CONTINUOUS, name="t")

    # -----------------------------
    # MTZ capacity constraints
    # -----------------------------
    for i, demand_i in enumerate(demands):
        for j, demand_j in enumerate(demands):
            if i != j and demand_i > 0 and demand_j > 0 and (i, j) in arc_list:
                m.addConstr(
                    u[j] - u[i] >= demand_j - vehicle_capacity * (1 - x[i, j]),
                    name=f"MTZ1_{i}_{j}",
                )

    for j, demand_j in enumerate(demands):
        if demand_j > 0:
            m.addConstr(u[j] >= demand_j, name=f"MTZ2_lb{j}")
            m.addConstr(u[j] <= vehicle_capacity, name=f"MTZ2_ub{j}")

    # -----------------------------
    # Vehicle count at depot
    # -----------------------------
    depot_idx = None
    for i, demand in enumerate(demands):
        if demand == 0:
            m.addConstr(
                gp.quicksum(
                    x[i, j]
                    for j in range(len(demands))
                    if (i, j) in x and i != j
                )
                <= nb_vehicles,
                name=f"CV{i}",
            )
            depot_idx = i
            break

    if depot_idx is None:
        raise ValueError("No depot found (expected a node with demand == 0).")

    # -----------------------------
    # Degree constraints (same)
    # -----------------------------
    for i, demand in enumerate(demands):
        if demand > 0:
            m.addConstr(
                gp.quicksum(
                    x[i, j]
                    for j in range(len(demands))
                    if (i, j) in x and i != j
                )
                == 1,
                name=f"LC{i}",
            )

    for j, demand in enumerate(demands):
        if demand > 0:
            m.addConstr(
                gp.quicksum(
                    x[i, j]
                    for i in range(len(demands))
                    if (i, j) in x and i != j
                )
                == 1,
                name=f"SC{j}",
            )

    # -----------------------------
    # CHANGE (TW): Time window bounds  (your (5))
    # ready_i <= t[i] <= due_i
    # -----------------------------
    for i in range(len(demands)):
        # allow depot too (often [0, horizon])
        m.addConstr(t[i] >= float(ready_time[i]), name=f"TW_lb{i}")
        m.addConstr(t[i] <= float(due_time[i]), name=f"TW_ub{i}")

    # Fix depot time to 0:
    m.addConstr(t[depot_idx] == float(ready_time[depot_idx]), name="DepotTimeFix")

    # -----------------------------
    # CHANGE (TW): Time propagation  (your (6))
    # t[j] >= t[i] + service_i + travel_ij - M*(1-x[i,j])
    # -----------------------------
    # A safe global Big-M:
    # when x=0, constraint should not bind even at extremes.
    max_due = float(max(due_time))
    max_service = float(max(service_time))
    max_travel = float(max(arc_costs)) if len(arc_costs) > 0 else 0.0
    M = max_due + max_service + max_travel

    for (i, j) in arc_list:
        if i == j:
            continue
        if j == depot_idx:
            continue
        travel_ij = float(cost_dict[(i, j)])
        m.addConstr(
            t[j] >= t[i] + float(service_time[i]) + travel_ij - M * (1 - x[i, j]),
            name=f"TWprop_{i}_{j}",
        )

    return m, x, u, t



def cvrp_subset_connections(demands, arc_index, arc_costs, nb_vehicles,
                            vehicle_capacity, connections, relax = False):
    """Generate Gurobi model for the TP.

    Parameters
    ----------
    demands: 1D np.array or list
        A list of customer demands, a null demand corresponds to the depot.
    arc_index : 2D np.array
        A list of arcs with source and destination
    costs: 2D np.array
        Costs for using arc from  i to j.
    nb_vehicles: int
        Number of vehicles.
    vehicle_capacity : int or float
        Uniform capacity for the vehicles.
    connections: 1D np.array
        Binary list indicating whether connection should be included.
    relax : bool, optional
        Indicate whether the cost variables should be linearized. Default is False.


    Returns
    -------
    m: gp.Model
        Gurobi model of TP instance.
    x: dict
        Dictionary of arc activation variables, keyed by (node.id(i), node.id(j)).
    u: dict
        Dictionnary of used capacity after delivering city j, keyed by node.id(j).

    """

    m = gp.Model()


    cost_dict = {(int(src), int(dst)): arc_costs[k]
             for k, (src, dst) in enumerate(zip(arc_index[0], arc_index[1]))
             if connections[k]}

    arc_list = [
        (int(i), int(j))
        for idx, (i, j) in enumerate(zip(arc_index[0], arc_index[1]))
        if connections[idx]
    ]

    # The activation variables tell whether an arc is used to travel from a place to another
    if relax:
        x = m.addVars(arc_list, obj=cost_dict, vtype=gp.GRB.CONTINUOUS, name="x")
    else:
        x = m.addVars(arc_list, obj=cost_dict, vtype=gp.GRB.BINARY, name="x")

    u = m.addVars(len(demands), vtype=gp.GRB.CONTINUOUS, name="u")

    # For each pair of customers, first MTZ inequality
    for i, demand_i in enumerate(demands):
        for j, demand_j in enumerate(demands):
            if(i!=j and demand_i > 0 and demand_j > 0 and (i, j) in x):
                m.addConstr(u[j] - u[i] >= demand_j - vehicle_capacity*(1 - x[i, j]),
                            name = f"MTZ1{i,j}")

    # For each customer, second MTZ inequality
    for j, demand_j in enumerate(demands):
        if(demand_j > 0):
            m.addConstr(u[j] >= demand_j, name=f"MTZ2_lb{j}")
            m.addConstr(u[j] <= vehicle_capacity, name=f"MTZ2_ub{j}")

    # Constraint for the number of vehicles leaving the depot
    for i, demand in enumerate(demands):
        if(demand == 0): 
            m.addConstr(
                gp.quicksum(x[i, j] for j in range(len(demands)) 
                            if (i, j) in x and i != j) <= nb_vehicles,
                name=f"CV{i}",
            )
            break
    # Leaving customer constraint
    for i, demand in enumerate(demands):
        if(demand > 0):
            m.addConstr(
                gp.quicksum(x[i, j] for j in range(len(demands))
                            if (i, j) in x and i != j) == 1,
                name=f"LC{i}"
            )
    
    #Serving customer constraint
    for j, demand in enumerate(demands):
        if(demand > 0):
            m.addConstr(
                gp.quicksum(x[i, j] for i in range(len(demands))
                            if (i, j) in x and i != j) == 1,
                name=f"SC{j}"
            )

    return m, x, u


def cvrptw_subset_connections(
    demands,
    arc_index,
    arc_costs,
    nb_vehicles,
    vehicle_capacity,
    connections,
    ready_time,
    due_time,
    service_time,
    relax=False,
):
    """Generate Gurobi model for the CVRPTW with a subset of arcs.

    Combines arc filtering from cvrp_subset_connections with time window
    constraints from cvrptw.

    Parameters
    ----------
    demands : 1D np.array or list
        Customer demands. A zero demand corresponds to the depot.
    arc_index : 2D np.array
        List of arcs with source and destination.
    arc_costs : 1D np.array
        Cost for using each arc.
    nb_vehicles : int
        Number of vehicles.
    vehicle_capacity : int or float
        Uniform capacity for the vehicles.
    connections : 1D np.array
        Binary list indicating whether each arc should be included.
    ready_time : 1D np.array
        Earliest service start time a_i for each node i.
    due_time : 1D np.array
        Latest service start time b_i for each node i.
    service_time : 1D np.array
        Service duration s_i for each node i.
    relax : bool, optional
        Whether to relax binary variables to continuous. Default is False.

    Returns
    -------
    m : gp.Model
        Gurobi model.
    x : dict
        Arc activation variables.
    u : dict
        MTZ capacity variables.
    t : dict
        Service start time variables.
    """

    m = gp.Model()

    # Filter arcs by connections mask
    cost_dict = {
        (int(src), int(dst)): arc_costs[k]
        for k, (src, dst) in enumerate(zip(arc_index[0], arc_index[1]))
        if connections[k]
    }

    arc_list = [
        (int(i), int(j))
        for idx, (i, j) in enumerate(zip(arc_index[0], arc_index[1]))
        if connections[idx]
    ]

    # Arc activation variables
    if relax:
        x = m.addVars(arc_list, obj=cost_dict, vtype=gp.GRB.CONTINUOUS, name="x")
    else:
        x = m.addVars(arc_list, obj=cost_dict, vtype=gp.GRB.BINARY, name="x")

    # MTZ capacity variables
    u = m.addVars(len(demands), vtype=gp.GRB.CONTINUOUS, name="u")

    # Time variables
    t = m.addVars(len(demands), vtype=gp.GRB.CONTINUOUS, name="t")

    # MTZ capacity constraints
    for i, demand_i in enumerate(demands):
        for j, demand_j in enumerate(demands):
            if i != j and demand_i > 0 and demand_j > 0 and (i, j) in x:
                m.addConstr(
                    u[j] - u[i] >= demand_j - vehicle_capacity * (1 - x[i, j]),
                    name=f"MTZ1{i, j}",
                )

    for j, demand_j in enumerate(demands):
        if demand_j > 0:
            m.addConstr(u[j] >= demand_j, name=f"MTZ2_lb{j}")
            m.addConstr(u[j] <= vehicle_capacity, name=f"MTZ2_ub{j}")

    # Vehicle count at depot
    depot_idx = None
    for i, demand in enumerate(demands):
        if demand == 0:
            m.addConstr(
                gp.quicksum(
                    x[i, j]
                    for j in range(len(demands))
                    if (i, j) in x and i != j
                )
                <= nb_vehicles,
                name=f"CV{i}",
            )
            depot_idx = i
            break

    if depot_idx is None:
        raise ValueError("No depot found (expected a node with demand == 0).")

    # Degree constraints
    for i, demand in enumerate(demands):
        if demand > 0:
            m.addConstr(
                gp.quicksum(
                    x[i, j]
                    for j in range(len(demands))
                    if (i, j) in x and i != j
                )
                == 1,
                name=f"LC{i}",
            )

    for j, demand in enumerate(demands):
        if demand > 0:
            m.addConstr(
                gp.quicksum(
                    x[i, j]
                    for i in range(len(demands))
                    if (i, j) in x and i != j
                )
                == 1,
                name=f"SC{j}",
            )

    # Time window bounds
    for i in range(len(demands)):
        m.addConstr(t[i] >= float(ready_time[i]), name=f"TW_lb{i}")
        m.addConstr(t[i] <= float(due_time[i]), name=f"TW_ub{i}")

    # Fix depot time
    m.addConstr(t[depot_idx] == float(ready_time[depot_idx]), name="DepotTimeFix")

    # Time propagation with Big-M
    max_due = float(max(due_time))
    max_service = float(max(service_time))
    max_travel = float(max(arc_costs)) if len(arc_costs) > 0 else 0.0
    M = max_due + max_service + max_travel

    for (i, j) in arc_list:
        if i == j:
            continue
        if j == depot_idx:
            continue
        travel_ij = float(cost_dict[(i, j)])
        m.addConstr(
            t[j] >= t[i] + float(service_time[i]) + travel_ij - M * (1 - x[i, j]),
            name=f"TWprop_{i}_{j}",
        )

    return m, x, u, t


def cvrp_via_VRP_Easy(demands, arc_index, arc_costs, nb_vehicles,
                            vehicle_capacity, connections):
    

    cost_dict = {(int(src), int(dst)): arc_costs[k]
             for k, (src, dst) in enumerate(zip(arc_index[0], arc_index[1]))
             if connections[k]}

    arc_list = [
        (int(i), int(j))
        for idx, (i, j) in enumerate(zip(arc_index[0], arc_index[1]))
        if connections[idx]
    ]
    
    model = solver.Model()

    # add vehicle type
    model.add_vehicle_type(id=1,
                           start_point_id=0,
                           end_point_id=0,
                           max_number=nb_vehicles,
                           capacity=vehicle_capacity,
                           var_cost_dist=1
                           )
    # add depot
    model.add_depot(id=0)

    for i in range(1, len(demands)):
        model.add_customer(id=i,
                           demand=int(demands[i])
                           )
        
    for arc in arc_list:
        model.add_link(start_point_id=arc[0],
                       end_point_id=arc[1],
                       distance=float(cost_dict[arc])
                       )
        
    model.set_parameters(time_limit=10)

    ''' If you have cplex 22.1 installed on your laptop windows you have to specify
        solver path'''
    
    # solver_path = r"C:\Users\bapti\IBM_CPLEX\cplex\bin\x64_win64\cplex2211.dll"

    
    # model.parameters.cplex_path = solver_path
    
    model.solve()

    if model.solution.is_defined :
        print(f"""Statistics :
        best lower bound : { model.statistics.best_lb } 
        
        solution time : {model.statistics.solution_time}
        
        number of nodes : {model.statistics.nb_branch_and_bound_nodes}
        
        solution value : {model.solution.value}

        root lower bound : {model.statistics.root_lb}

        root root time : {model.statistics.root_time}.
        """)
        print(f"Status : {model.status}.\n")
        print(f"Message : {model.message}.\n")   
        for route in model.solution.routes:            
            print(f"Vehicle Type id : {route.vehicle_type_id}.")
            print(f"Ids : {route.point_ids}.")
            print(f"Load : {route.cap_consumption}.\n")

    
from gurobipy import GRB
def solve_relaxed_flow_cvrp( demands, arc_index, arc_costs, nb_vehicles,
 vehicle_capacity, connections, time_limit=100, verbose=False ):
    
    """ Gurobi version of relaxed CVRP flow formulation. Builds, solves,
    and returns only the solution dictionary. 
    Returns ------- solution : dict {(i,j): y_ij_value}
    Empty dict if no feasible solution was found. """ 
  
    m = gp.Model("relaxed_CVRP_flow")

    m.setParam("OutputFlag", 0)

    m.Params.TimeLimit = time_limit

    arc_list = [ (int(i), int(j)) for k, (i, j) in enumerate(zip(arc_index[0], arc_index[1])) if connections[k] ]

    cost_dict = { (int(i), int(j)): float(arc_costs[k]) for k, (i, j) in enumerate(zip(arc_index[0], arc_index[1])) if connections[k] }

    depot_candidates = [i for i, d in enumerate(demands) if d == 0]

    if len(depot_candidates) != 1:
         raise ValueError("Demands must contain exactly one depot (one zero-demand entry).")
    
    depot = depot_candidates[0]

    n = len(demands)

    y = {(i, j): m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"y_{i}_{j}") for (i, j) in arc_list}
    f = {(i, j): m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"f_{i}_{j}") for (i, j) in arc_list} 

    m.setObjective(gp.quicksum(cost_dict[i, j] * y[i, j] for (i, j) in arc_list), GRB.MINIMIZE) 

    for i in range(n):
       if i != depot:
         m.addConstr(gp.quicksum(y[i, j] for j in range(n) if (i, j) in y) == 1, name=f"out_{i}")

    for j in range(n):
        if j != depot:
            m.addConstr(gp.quicksum(y[i, j] for i in range(n) if (i, j) in y) == 1, name=f"in_{j}")
            m.addConstr(gp.quicksum(y[depot, j] for j in range(n) if (depot, j) in y) <= nb_vehicles, name="veh_limit")

    for (i, j) in arc_list:
        m.addConstr(f[i, j] <= vehicle_capacity * y[i, j], name=f"cap_{i}_{j}")

    for i in range(n):
         if i != depot:
             outgoing = gp.quicksum(f[i, j] for j in range(n) if (i, j) in f)
             incoming = gp.quicksum(f[j, i] for j in range(n) if (j, i) in f)
             m.addConstr(outgoing - incoming == float(demands[i]), name=f"flow_{i}")
    m.optimize() 
    if m.Status != GRB.OPTIMAL and m.Status != GRB.TIME_LIMIT: 
        return {} 
    solution = {(i, j): y[i, j].X for (i, j) in y} 
    return solution
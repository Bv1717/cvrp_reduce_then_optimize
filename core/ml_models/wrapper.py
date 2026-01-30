# """Wrapper and helper functions for ML-based FCTP algorithms. """

##Changes
"""Wrapper and helper functions for ML-based CVRP algorithms. """
##End changes

import time

import numpy as np
import torch
from torch_geometric.data import Data
import gurobipy as gp
import matplotlib.pyplot as plt
import networkx as nx


# from core.utils.cvrp import CapacitatedFCTP
# from core.utils.cvrp import FixedStepFCTP
# from core.fctp_solvers.ip_grb import get_fctp_bfs
# from core.fctp_solvers.ip_grb import fctp_subset_connections
# from core.fctp_solvers.ip_grb import capacitated_fctp_subset_connections
# from core.fctp_solvers.ip_grb import fixed_step_fctp_subset_connections
# from core.fctp_solvers.ip_grb import capacitated_fctp
# from core.fctp_solvers.ip_grb import sol_vals
# from core.fctp_solvers.heuristics import lcm_fctp
# from core.utils.preprocessing import get_subproblems

##Changes
from core.cvrp_solvers.ip_grb import cvrp_subset_connections
from core.cvrp_solvers.ip_grb import sol_vals
from core.cvrp_solvers.heuristics import Clark_Wright_heuristic
from core.cvrp_solvers.heuristics import heu_solve_HGS_VRP
from core.cvrp_solvers.ip_grb import cvrp_via_VRP_Easy
##End changes




# from core.utils.postprocessing import reindex_sol


# def get_max_likelihood_sol(instance, predictions):
#     """Get feasibile solution that greedily maximizes the total edge likelihood.

#     FCTP: Apply Least-Cost Method using negative predictions as costs.
#     C-FCTP: Solve Fixed-Charge Problem using inverse predictions as fixed costs.

#     Parameters
#     ----------
#     instance: FCTP
#         FCTP instance.
#     predictions: 2D np.array
#         Prediction matrix.

#     Returns
#     -------
#     sol: dict
#         Greedy max-likelihood solution.

#     """
#     if isinstance(instance, CapacitatedFCTP):
#         m, x, _ = capacitated_fctp(
#             supplies=instance.supply,
#             demands=instance.demand,
#             var_costs=np.zeros_like(predictions, dtype=int),
#             fix_costs=1 / predictions,
#             edge_capacities=instance.edge_capacities,
#         )
#         m.setParam("OutputFlag", 0)
#         m.setParam("TimeLimit", 10)
#         m.setParam("Seed", 0)
#         m.optimize()
#         sol = sol_vals(x)
#     else:
#         sol = lcm_fctp(instance, costs=-predictions)
#     return sol


# def sol_edge_predictor_wrapper(instance, predictor_model):
#     """Wrapper function for solution edge prediction.

#     Parameters
#     ----------
#     instance: FCTP
#         FCTP instance.
#     predictor_model: BaseSolEdgePredictor
#         Solution edge predictor.

#     Returns
#     -------
#     predictions: 2D np.array
#         Prediction matrix.

#     """
#     predictor_model, feature_fun = predictor_model

#     inputs = tuple(feature_fun(instance))
#     inputs = tuple([torch.tensor(input) for input in inputs])

#     predictions = predictor_model.predict_edges(inputs, train=False)
#     if isinstance(predictions, tuple):
#         _, predictions = predictions

#     predictions = predictions.detach().numpy()

#     return predictions.reshape(instance.m, instance.n)


# def get_reduced_problem(
#     instance,
#     predictor_model,
#     threshold_type="size",
#     threshold=0.5,
# ):
#     """Wrapper function to get reduced problem.

#     Step 1: Make predictions
#     Step 2: Select edges (incl. feasibiliyt edges)

#     Parameters
#     ----------
#     instance: FCTP
#         FCTP instance.
#     predictor_model: BaseSolEdgePredictor
#         Solution edge predictor.
#     threshold_type: str, optional
#         Type of threshold. Can be 'size' or 'prob'. Default is 'size'.
#     threshold: float, optional
#         Threshold value. Default is 0.5.

#     Returns
#     -------
#     relevant_connections: 2D np.array
#         Binary edge mask indicating selected edges
#     tuple of int:
#         Number of predicted edges (without feasibility edges) and selected edges.

#     """
#     # get edge values/likelihoods
#     edge_likelihood = sol_edge_predictor_wrapper(instance, predictor_model)

#     # select the most likely edges
#     if threshold_type == "size":
#         threshold = np.quantile(edge_likelihood, 1 - threshold)
#     relevant_connections = edge_likelihood >= threshold

#     num_edges_pred = np.sum(relevant_connections)

#     # add heuristic solution to set of edges to guarantee feasibility
#     greedy_sol = get_max_likelihood_sol(instance, edge_likelihood)
#     for (i, j), val in greedy_sol.items():
#         if val > 0:
#             relevant_connections[i, j] = True

#     num_edges_enriched = np.sum(relevant_connections)

#     return relevant_connections, (num_edges_pred, num_edges_enriched)


# def solve_reduced_problem(
#     instance,
#     relevant_connections,
#     decoder="exact",
#     decoder_cfg=None,
#     decoder_env=None,
#     seed=0,
# ):
#     """Wrapper function to solve reduced problem.

#     Parameters
#     ----------
#     instance: FCTP
#         FCTP instance.
#     relevant_connections: 2D np.array
#         Binary edge mask indicating selected edges
#     decoder: str, optional
#         Solver to use. Can be 'exact', 'lp', 'ts' or 'ea'. Default is 'exact'.
#     decoder_cfg: dict, optional
#         Decoder config.
#     decoder_env: optional
#         Decoder environment (for TS or EA).
#     seed: int, optional
#         Solver seed. Default is 0.

#     Returns
#     -------
#     solution: dict
#         Solution dictionary.
#     runtime: int or float
#         Solver runtime.
#     status: int or str
#         Solver status code.
#     mip_gap: float
#         MIP gap if applicable.

#     """
#     if decoder_cfg is None:
#         decoder_cfg = {}

#     # solve reduced FCTP
#     status = None
#     mip_gap = None
#     if decoder in ["exact", "lp"]:
#         if isinstance(instance, CapacitatedFCTP):
#             model, x, _ = capacitated_fctp_subset_connections(
#                 instance.supply,
#                 instance.demand,
#                 instance.var_costs,
#                 instance.fix_costs,
#                 instance.edge_capacities,
#                 relevant_connections,
#                 relax=(decoder == "lp"),
#             )
#         elif isinstance(instance, FixedStepFCTP):
#             model, x, _ = fixed_step_fctp_subset_connections(
#                 instance.supply,
#                 instance.demand,
#                 instance.var_costs,
#                 instance.fix_costs,
#                 instance.vehicle_capacities,
#                 relevant_connections,
#                 relax=(decoder == "lp"),
#             )
#         else:
#             model, x, _ = fctp_subset_connections(
#                 instance.supply,
#                 instance.demand,
#                 instance.var_costs,
#                 instance.fix_costs,
#                 relevant_connections,
#                 relax=(decoder == "lp"),
#             )
#         model.setParam("OutputFlag", 0)
#         if decoder_cfg.get("grb_timeout") is not None:
#             model.setParam("TimeLimit", decoder_cfg["grb_timeout"])
#         if decoder_cfg.get("grb_threads") is not None:
#             model.setParam("Threads", decoder_cfg.get("grb_threads"))
#         model.setParam("Seed", seed)
#         model.optimize()
#         solution = sol_vals(x)
#         runtime = model.Runtime
#         status = model.Status
#         if decoder == "exact":
#             mip_gap = model.MIPGap
#     elif decoder == "ts":
#         if isinstance(instance, CapacitatedFCTP) or isinstance(instance, FixedStepFCTP):
#             raise NotImplementedError
#         start = time.time()
#         subproblems = get_subproblems(instance, relevant_connections)
#         if len(subproblems) == 1:
#             bfs = get_fctp_bfs(instance, edge_mask=relevant_connections)
#             solution, _, _ = decoder_env.run(
#                 instance,
#                 bfs,
#                 decoder_cfg,
#                 relevant_connections,
#             )
#         else:
#             sub_sols = []
#             for subproblem in subproblems:
#                 (
#                     s_nodes,
#                     d_nodes,
#                     sub_instance,
#                     sub_conns,
#                 ) = subproblem
#                 if len(s_nodes) == 1:
#                     sub_sol = {(0, j): d for j, d in enumerate(sub_instance.demand)}
#                 elif len(d_nodes) == 1:
#                     sub_sol = {(i, 0): s for i, s in enumerate(sub_instance.supply)}
#                 else:
#                     bfs = get_fctp_bfs(
#                         sub_instance,
#                         edge_mask=sub_conns,
#                     )
#                     sub_sol, _, _ = decoder_env.run(
#                         sub_instance,
#                         bfs,
#                         decoder_cfg,
#                         sub_conns,
#                     )
#                 sub_sols.append(reindex_sol(sub_sol, s_nodes, d_nodes))
#             solution = {k: v for sub_sol in sub_sols for (k, v) in sub_sol.items()}
#         runtime = time.time() - start
#     elif decoder == "ea":
#         if isinstance(instance, CapacitatedFCTP) or isinstance(instance, FixedStepFCTP):
#             raise NotImplementedError
#         start = time.time()
#         subproblems = get_subproblems(instance, relevant_connections)
#         if len(subproblems) == 1:
#             solution, _, _ = decoder_env.run(
#                 instance,
#                 decoder_cfg,
#                 relevant_connections,
#             )
#         else:
#             sub_sols = []
#             for subproblem in subproblems:
#                 (
#                     s_nodes,
#                     d_nodes,
#                     sub_instance,
#                     sub_conns,
#                 ) = subproblem
#                 if len(s_nodes) == 1:
#                     sub_sol = {(0, j): d for j, d in enumerate(sub_instance.demand)}
#                 elif len(d_nodes) == 1:
#                     sub_sol = {(i, 0): s for i, s in enumerate(sub_instance.supply)}
#                 else:
#                     sub_sol, _, _ = decoder_env.run(
#                         sub_instance,
#                         decoder_cfg,
#                         sub_conns,
#                     )
#                 sub_sols.append(reindex_sol(sub_sol, s_nodes, d_nodes))
#             solution = {k: v for sub_sol in sub_sols for (k, v) in sub_sol.items()}
#         runtime = time.time() - start
#     else:
#         raise ValueError
#     return solution, runtime, status, mip_gap


# def ml_based_fctp_reduction(
#     instance,
#     predictor_model,
#     threshold_type="size",
#     threshold=0.5,
#     decoder="exact",
#     decoder_cfg=None,
#     decoder_env=None,
#     seed=0,
# ):
#     """Wrapper function for ML-based reduce-then-optimize.

#     Step 1: Get reduced instance
#     Step 2: Solve reduced instance

#     Parameters
#     ----------
#     instance: FCTP
#         FCTP instance.
#     predictor_model: BaseSolEdgePredictor
#         Solution edge predictor.
#     threshold_type: str, optional
#         Type of threshold. Can be 'size' or 'prob'. Default is 'size'.
#     threshold: float, optional
#         Threshold value. Default is 0.5.
#     decoder: str, optional
#         Solver to use. Can be 'exact', 'lp', 'ts' or 'ea'. Default is 'exact'.
#     decoder_cfg: dict, optional
#         Decoder config.
#     decoder_env: optional
#         Decoder environment (for TS or EA).
#     seed: int, optional
#         Solver seed. Default is 0.

#     Returns
#     -------
#     solution: dict
#         Solution dictionary.
#     num_edges_pred:
#         Number of predicted edges (without feasibility edges)
#     num_edges_enriched:
#         Number of selected edges.
#     runtime: int or float
#         Solver runtime.
#     status: int or str
#         Solver status code.
#     mip_gap: float
#         MIP gap if applicable.

#     """

#     relevant_connections, (num_edges_pred, num_edges_enriched) = get_reduced_problem(
#         instance,
#         predictor_model,
#         threshold_type,
#         threshold,
#     )

#     solution, runtime, status, mip_gap = solve_reduced_problem(
#         instance,
#         relevant_connections,
#         decoder,
#         decoder_cfg,
#         decoder_env,
#         seed,
#     )

#     return solution, num_edges_pred, num_edges_enriched, runtime, status, mip_gap


##Changes 


def plot_edge_heatmap(instance, arc_index, arc_likelihood, cmap="viridis"):
    # Build graph
    G = nx.DiGraph()  # or Graph() if undirected

    # Add nodes with coordinates
    for i, node in enumerate(instance.nodes):
        G.add_node(i, pos=(node.x, node.y))


    # Add edges with weights
    src, dst = arc_index
    for i in range(len(src)):
        G.add_edge(int(src[i]), int(dst[i]), weight=float(arc_likelihood[i]))

    # Extract positions
    pos = nx.get_node_attributes(G, "pos")

    # Normalize weights for colormap
    weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
    norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color="black")

    # Draw edges with colormap
    edges = G.edges()
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges,
        edge_color=norm,
        edge_cmap=plt.cm.get_cmap(cmap),
        width=1 + 3 * norm,  # thicker for high weight
        alpha=0.8,
    )

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=weights.min(), vmax=weights.max()))
    sm.set_array([])
    plt.colorbar(sm, label="Predicted edge weight")

    plt.axis("equal")
    plt.title("GNN Edge Likelihood Heatmap")
    plt.show()



def plot_top_k_edges(instance, arc_index, arc_likelihood, k, cmap="viridis"):
    """
    Draw only the top-k edges according to arc_likelihood.
    """

    # Convert to numpy
    weights = np.array(arc_likelihood).reshape(-1)
    src, dst = arc_index

    # ---- 1. Select top-k edges ----
    if k < 1:  
        # interpret k as a fraction (e.g., 0.1 = top 10%)
        k = int(len(weights) * k)

    k = max(1, min(k, len(weights)))  # clamp


    top_idx = np.argsort(weights)[-k:]  # indices of top-k edges

    # ---- 2. Build graph with only top-k edges ----
    G = nx.DiGraph()

    # Add nodes with coordinates
    for i, node in enumerate(instance.nodes):
        G.add_node(i, pos=(node.x, node.y))

    # Add only selected edges
    for i in top_idx:
        G.add_edge(int(src[i]), int(dst[i]), weight=float(weights[i]))

    # ---- 3. Extract positions and normalized weights ----
    pos = nx.get_node_attributes(G, "pos")
    edge_weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
    norm = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min() + 1e-9)

    # ---- 4. Draw ----
    plt.figure(figsize=(8, 8))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color="black")

    # Draw edges with colormap
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=G.edges(),
        edge_color=norm,
        edge_cmap=plt.cm.get_cmap(cmap),
        width=1 + 3 * norm,
        alpha=0.9,
    )

    

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=edge_weights.min(), vmax=edge_weights.max()))
    sm.set_array([])
    plt.colorbar(sm, label="Predicted edge weight")

    plt.axis("equal")
    plt.title(f"Top-{k} Predicted Edges (Heatmap)")
    plt.show()






def get_max_likelihood_sol(instance, relevant_connections, prediction):
    """Get feasible solution that greedily maximizes the total edge likelihood.

    CVRP: Apply Clark et Wright heuristic using handmade costs obtained from predictions

    Parameters
    ----------
    instance: CVRP
        CVRP instance.
    relevant_connections: 1D np.array
        Prediction list.

    Returns
    -------
    sol: dict
        Handmade heuristic_costs solution.

    """

    # select the most likely arcs

    handmade_costs = [0]*len(relevant_connections)

    for i in range(len(relevant_connections)):
        if(relevant_connections[i]):
            handmade_costs[i] = 1/(prediction[i] + 0.001)
        else:
            handmade_costs[i] = 100*(1/(prediction[i]+ 0.001))


    all_connections = [True]*len(handmade_costs)
    sol, _ = heu_solve_HGS_VRP(instance.demands, instance.arc_index, handmade_costs, 
                            instance.nb_vehicles, instance.vehicle_capacity, all_connections,
                            heu_time=5)
    # sol = Clark_Wright_heuristic(instance.demands, instance.arc_index,
    #                             handmade_costs, instance.nb_vehicles,
    #                             instance.vehicle_capacity)
    return sol


def sol_arc_predictor_wrapper(instance, predictor_model):
    """Wrapper function for solution arc prediction.

    Parameters
    ----------
    instance: CVRP
        CVRP instance.
    predictor_model: BaseSolArcPredictor
        Solution arc predictor.

    Returns
    -------
    predictions: np array
        Array of arc predictions aligned with arc_index.
    arc_index: np.array
        2 x E array of source/destination indices for each arc.

    """
    predictor_model, feature_fun = predictor_model

    inputs = tuple(feature_fun(instance))
    inputs = tuple([torch.tensor(input) for input in inputs])

    x = torch.as_tensor(inputs[0])
    arc_attr = torch.as_tensor(inputs[1])
    arc_index_input = torch.as_tensor(inputs[2],  dtype=torch.long)

    data = Data(x=x, edge_index=arc_index_input, edge_attr=arc_attr)

    predictions = predictor_model.predict_arcs(data, train=False)
    
    if isinstance(predictions, tuple):
        _, predictions = predictions


    predictions = predictions.detach().numpy()

    arc_index = inputs[2].numpy()  # assuming feature_fun returns arc_index as third element

    return predictions, arc_index

def get_reduced_problem(
    instance,
    predictor_model,
    test_list=None,
    threshold_type="size",
    threshold=0.5,
):
    """Wrapper function to get reduced problem.

    Step 1: Make predictions
    Step 2: Select edges (incl. feasibiliyt edges)

    Parameters
    ----------
    instance: CVRP
        CVRP instance.
    predictor_model: BaseSolArcPredictor
        Solution arc predictor.
    threshold_type: str, optional
        Type of threshold. Can be 'size' or 'prob'. Default is 'size'.
    threshold: float, optional
        Threshold value. Default is 0.5.

    Returns
    -------
    relevant_connections: 1D np.array
        Binary arc mask indicating selected arcs
    arc_index: 2D np array
        List of arcs with source and destination
    tuple of int:
        Number of predicted arcs (without feasibility arcs) and selected arcs.

    """
    # get arc values/likelihoods
    arc_likelihood, arc_index = sol_arc_predictor_wrapper(instance, predictor_model)


    # plot_edge_heatmap(instance, arc_index, arc_likelihood)

    # plot_top_k_edges(instance, arc_index, arc_likelihood, k=1000)


    # select the most likely arcs
    if threshold_type == "size":
        threshold = np.quantile(arc_likelihood, 1 - threshold)
    relevant_connections = arc_likelihood >= threshold
    relevant_connections = relevant_connections.reshape(-1) 

    

    num_arcs_pred = np.sum(relevant_connections)

    arc_index_map = {
        (int(instance.arc_index[0, idx]), int(instance.arc_index[1, idx])): idx
        for idx in range(instance.arc_index.shape[1])
    }

    # add heuristic solution to set of edges to guarantee feasibility
    greedy_sol = get_max_likelihood_sol(instance, relevant_connections, arc_likelihood.reshape(-1))
    for i, val in greedy_sol.items():
        if val > 0:
            relevant_connections[arc_index_map[i]] = True

    num_arcs_enriched = np.sum(relevant_connections)

    print("num_arc_preds : ", num_arcs_pred, "num_arc_added : ", num_arcs_enriched-num_arcs_pred)

    return relevant_connections, arc_index, (num_arcs_pred, num_arcs_enriched)

def solve_reduced_problem(
    instance,
    arc_index,
    relevant_connections,
    decoder="exact",
    decoder_cfg=None,
    decoder_env=None,
    seed=0,
    HGS_run_time=100
):
    """Wrapper function to solve reduced problem.

    Parameters
    ----------
    instance: CVRP
        CVRP instance.
    arc_index : 2D np.array
        List of arcs with source and destination
    relevant_connections: 1D np.array
        Binary arc mask indicating selected arcs.
    decoder: str, optional
        Solver to use. Can be 'exact', 'lp'. Default is 'exact'.
    decoder_cfg: dict, optional
        Decoder config.
    decoder_env: optional
        Decoder environment (for TS or EA).
    seed: int, optional
        Solver seed. Default is 0.

    Returns
    -------
    solution: dict
        Solution dictionary.
    runtime: int or float
        Solver runtime.
    status: int or str
        Solver status code.
    mip_gap: float
        MIP gap if applicable.

    """
    if decoder_cfg is None:
        decoder_cfg = {}

    status = None
    mip_gap = None
    if decoder in ["exact"]: 

        # model, x, _ = cvrp_subset_connections(
        #     instance.demands,
        #     instance.arc_index,
        #     instance.arc_costs,
        #     instance.nb_vehicles,
        #     instance.vehicle_capacity,
        #     relevant_connections,
        #     relax=False,
        # )
        # model.setParam("OutputFlag", 1)
        # if decoder_cfg.get("grb_timeout") is not None:
        #     model.setParam("TimeLimit", decoder_cfg["grb_timeout"])
        # if decoder_cfg.get("grb_threads") is not None:
        #     model.setParam("Threads", decoder_cfg.get("grb_threads"))
        # model.setParam("Seed", seed)
        # model.optimize()
        # if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.SUBOPTIMAL:
        #     solution = sol_vals(x)
        #     print("Optimal objective value with restriction:", model.objVal)
        # else:
        #     print("No feasible solution found, status:", model.status)
        #     solution = {}
        
        # runtime = model.Runtime
        # status = model.Status
        # mip_gap = model.MIPGap

        # solution, _ = heu_solve_HGS_VRP(instance.demands, instance.arc_index, 
        #               instance.arc_costs, instance.nb_vehicles,
        #                 instance.vehicle_capacity, relevant_connections, heu_time=HGS_run_time)
        solution,runtime, optimal_value  = cvrp_via_VRP_Easy(instance.demands, instance.arc_index, instance.arc_costs,
                                       instance.nb_vehicles, instance.vehicle_capacity, relevant_connections)
    return solution, runtime, status, optimal_value  #solution, runtime, status, mip_gap

def ml_based_cvrp_reduction(
    instance,
    predictor_model,
    test_list=None,
    threshold_type="size",
    threshold=0.5,
    decoder="exact",
    decoder_cfg=None,
    decoder_env=None,
    seed=0,
    HGS_run_time=100
):
    """Wrapper function for ML-based reduce-then-optimize.

    Step 1: Get reduced instance
    Step 2: Solve reduced instance

    Parameters
    ----------
    instance: CVRP
        CVRP instance.
    predictor_model: BaseSolArcPredictor
        Solution arc predictor.
    threshold_type: str, optional
        Type of threshold. Can be 'size' or 'prob'. Default is 'size'.
    threshold: float, optional
        Threshold value. Default is 0.5.
    decoder: str, optional
        Solver to use. Can be 'exact', 'lp'. Default is 'exact'.
    decoder_cfg: dict, optional
        Decoder config.
    decoder_env: optional
        Decoder environment (for TS or EA).
    seed: int, optional
        Solver seed. Default is 0.

    Returns
    -------
    solution: dict
        Solution dictionary.
    num_arcs_pred:
        Number of predicted arcs (without feasibility arcs)
    num_arcs_enriched:
        Number of selected arcs.
    runtime: int or float
        Solver runtime.
    status: int or str
        Solver status code.
    mip_gap: float
        MIP gap if applicable.

    """

    relevant_connections, arc_index, (num_arcs_pred, num_arcs_enriched) = get_reduced_problem(
        instance,
        predictor_model,
        test_list,
        threshold_type,
        threshold,
    )

    solution, runtime, status, optimal_value = solve_reduced_problem(
        instance,
        arc_index,
        relevant_connections,
        decoder,
        decoder_cfg,
        decoder_env,
        seed,
        HGS_run_time
    )

    return solution, num_arcs_pred, num_arcs_enriched, runtime, status, optimal_value

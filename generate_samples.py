import os
import gzip
import pickle as pkl
import numpy as np
import gurobipy as gp
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pyvrp import Model as HGS_Model
from pyvrp.stop import MaxRuntime as HGS_MaxRuntime
import re
import math
import hydra
from omegaconf import DictConfig
# from typing import List

from core.utils.cvrp import CVRP_node
from core.utils.cvrp import CVRP
from core.cvrp_solvers.ip_grb import cvrp
from core.cvrp_solvers.ip_grb import cvrptw
from core.cvrp_solvers.ip_grb import sol_vals
from core.cvrp_solvers.heuristics import Clark_Wright_heuristic
from core.evaluation.benchmarking_utils import get_mip_gap_by_method
from core.evaluation.benchmarking_utils import get_num_edges_by_method
from core.evaluation.benchmarking_utils import get_optgaps_by_method
from core.evaluation.benchmarking_utils import get_objval_by_method
from core.evaluation.benchmarking_utils import get_runtimes_by_method
from core.evaluation.benchmarking_utils import get_solver_runtimes_by_method
from core.evaluation.benchmarking_utils import get_num_arcs_used_by_method

from sklearn.metrics import ConfusionMatrixDisplay


from core.fctp_heuristics_julia.python_wrapper import Frank_Wolfe_regularisation_env

from core.cvrp_solvers.ip_grb import cvrp_via_VRP_Easy

# --- Generator function ---
def generate_cvrp_instance(num_nodes=20, vehicle_capacity=30, nb_vehicles=5):
    # Create nodes (node 0 = depot, others = clients)
    nodes = [CVRP_node(0, demand=0, x=0.0, y=0.0)]
    for i in range(1, num_nodes):
        demand = np.random.randint(1, 10)
        x, y = np.random.rand(2) * 100
        nodes.append(CVRP_node(i, demand, x, y))

    # Fully connected directed arcs (excluding self-loops)
    arc_index = np.array(
        [[i for i in range(num_nodes) for j in range(num_nodes) if i != j],
         [j for i in range(num_nodes) for j in range(num_nodes) if i != j]]
    )

    # Random arc costs
    arc_costs = np.random.randint(1, 20, size=arc_index.shape[1])

    return CVRP(nodes, arc_index, vehicle_capacity, arc_costs, nb_vehicles)

def generate_cvrp_instances_Munich(path_nodes: str, path_arcs:str, vehicle_capacity = 30, nb_vehicles = 5):
    df_nodes = pd.read_csv(path_nodes)
    indices = df_nodes["Index"].to_numpy()
    x_coords = df_nodes["X"].to_numpy()
    y_coords = df_nodes["Y"].to_numpy()
    demands = df_nodes["Demand/Capacity"].to_numpy()
    demands = [np.random.randint(1, 10)*(demands[i] != 0) for i in range(len(indices))]
    print(demands)

    # demands = np.array(demands)

    # plt.figure(figsize=(8, 6))

    # # Scatter plot, color points by demand
    # scatter = plt.scatter(x_coords, y_coords, c=demands, cmap="viridis", s=100, edgecolors="black")

    # # Annotate each point with its demand
    # for i in range(len(x_coords)):
    #     plt.annotate(str(demands[i]), (x_coords[i] + 0.05, y_coords[i] + 0.05), fontsize=9)

    # # Add colorbar to show demand scale
    # cbar = plt.colorbar(scatter)
    # cbar.set_label("Demand")

    # plt.title("Node Distribution with Demands")
    # plt.xlabel("X Coordinate")
    # plt.ylabel("Y Coordinate")
    # plt.grid(True)
    # plt.show()

    nodes = []
    for idx in indices: ##need to verify that only one node has 0 demand == depot
        nodes.append(CVRP_node(idx, demands[idx], x_coords[idx], y_coords[idx]))

    df_arcs = pd.read_csv(path_arcs, header = None, names = ["source", "target", "distance"])
    
    source = df_arcs["source"].to_numpy()
    target = df_arcs["target"].to_numpy()
    distance = df_arcs["distance"].to_numpy()
    arc_index_list = [[], []]
    arc_costs_list = []

    for i, src in enumerate(source):
        if src != target[i]:
            arc_index_list[0].append(src)
            arc_index_list[1].append(target[i])
            arc_costs_list.append(distance[i])
    
    arc_index = np.array(arc_index_list)
    arc_costs = np.array(arc_costs_list)
    return CVRP(nodes, arc_index, vehicle_capacity, arc_costs, nb_vehicles)


def generate_subset_cvrp_instances_Munich(path_nodes: str, path_arcs:str, vehicle_capacity = 30, nb_vehicles = 5):
    df_nodes = pd.read_csv(path_nodes)
    indices = df_nodes["Index"].to_numpy()
    x_coords = df_nodes["X"].to_numpy()
    y_coords = df_nodes["Y"].to_numpy()
    demands = df_nodes["Demand/Capacity"].to_numpy()
    demands = [np.random.randint(1, 10)*(demands[i] != 0) for i in range(len(indices))]
    print(demands)

    # demands = np.array(demands)

    # plt.figure(figsize=(8, 6))

    # # Scatter plot, color points by demand
    # scatter = plt.scatter(x_coords, y_coords, c=demands, cmap="viridis", s=100, edgecolors="black")

    # # Annotate each point with its demand
    # for i in range(len(x_coords)):
    #     plt.annotate(str(demands[i]), (x_coords[i] + 0.05, y_coords[i] + 0.05), fontsize=9)

    # # Add colorbar to show demand scale
    # cbar = plt.colorbar(scatter)
    # cbar.set_label("Demand")

    # plt.title("Node Distribution with Demands")
    # plt.xlabel("X Coordinate")
    # plt.ylabel("Y Coordinate")
    # plt.grid(True)
    # plt.show()

    nodes = []
    for idx in indices: ##need to verify that only one node has 0 demand == depot
        nodes.append(CVRP_node(idx, demands[idx], x_coords[idx], y_coords[idx]))

    df_arcs = pd.read_csv(path_arcs, header = None, names = ["source", "target", "distance"])
    
    source = df_arcs["source"].to_numpy()
    target = df_arcs["target"].to_numpy()
    distance = df_arcs["distance"].to_numpy()
    arc_index_list = [[], []]
    arc_costs_list = []

    for i, src in enumerate(source):
        if src != target[i]:
            arc_index_list[0].append(src)
            arc_index_list[1].append(target[i])
            arc_costs_list.append(distance[i])
    
    arc_index = np.array(arc_index_list)
    arc_costs = np.array(arc_costs_list)
    return CVRP(nodes, arc_index, vehicle_capacity, arc_costs, nb_vehicles)

#for tw generation
def assign_time_windows_simple(
    nodes,
    horizon=1000.0,
    width_min=150.0,
    width_max=300.0,
    service_time=0.0,
    seed=0,
):
    """
    Assign synthetic time windows to nodes for CVRPTW.

    Depot gets [0, horizon].
    Each customer gets [a_i, a_i + width] with width ~ U(width_min, width_max).

    This is a *data generator* helper: the solver must still enforce TW constraints.
    """
    rng = np.random.default_rng(seed)

    if width_min <= 0 or width_max < width_min:
        raise ValueError("Invalid time window width range.")

    # Identify depot (best effort)
    depot = None
    for n in nodes:
        if n.node_id == 0 or n.demand == 0:
            depot = n
            break

    if depot is not None:
        depot.ready_time = 0.0
        depot.due_time = float(horizon)
        depot.service_time = float(service_time)

    for n in nodes:
        if depot is not None and n is depot:
            continue

        n.service_time = float(service_time)

        w = float(rng.uniform(width_min, width_max))
        latest = max(0.0, float(horizon) - w)
        a = float(rng.uniform(0.0, latest))
        n.ready_time = a
        n.due_time = a + w

    return nodes


# def generate_restricted_cvrp_instances_Munich(path_nodes: str, path_arcs:str, vehicle_capacity = 30, nb_vehicles = 5,
#                                                      nb_clients = 20):
#     df_nodes = pd.read_csv(path_nodes)
#     indices = df_nodes["Index"].to_numpy()
#     x_coords = df_nodes["X"].to_numpy()
#     y_coords = df_nodes["Y"].to_numpy()
#     demands = df_nodes["Demand/Capacity"].to_numpy()
#     demands = [np.random.randint(1, 10)*(demands[i] != 0) for i in range(len(indices))]

#     demands = np.array(demands)

#     # plt.figure(figsize=(8, 6))

#     # # Scatter plot, color points by demand
#     # scatter = plt.scatter(x_coords, y_coords, c=demands, cmap="viridis", s=100, edgecolors="black")

#     # # Annotate each point with its demand
#     # for i in range(len(x_coords)):
#     #     plt.annotate(str(demands[i]), (x_coords[i] + 0.05, y_coords[i] + 0.05), fontsize=9)

#     # # Add colorbar to show demand scale
#     # cbar = plt.colorbar(scatter)
#     # cbar.set_label("Demand")

#     # plt.title("Node Distribution with Demands")
#     # plt.xlabel("X Coordinate")
#     # plt.ylabel("Y Coordinate")
#     # plt.grid(True)
#     # plt.show()

#     # Permute numbers 1..nb_clients
#     random_indices = np.random.permutation(np.arange(1, len(demands)))

#     # Prepend 0
#     indices = np.concatenate(([0], random_indices[:nb_clients]))

#     nodes = []
#     for i, idx in enumerate(indices): ##need to verify that only one node has 0 demand == depot
#         nodes.append(CVRP_node(i, demands[idx], x_coords[idx], y_coords[idx]))

#     df_arcs = pd.read_csv(path_arcs, header = None, names = ["source", "target", "distance"])
    
#     source = df_arcs["source"].to_numpy()
#     target = df_arcs["target"].to_numpy()
#     distance = df_arcs["distance"].to_numpy()
#     arc_index_list = [[], []]
#     arc_costs_list = []

#     for i, src in enumerate(source):
#         if src != target[i] and src in indices and target[i] in indices:
#             src_pos = np.where(indices == src)[0][0]
#             target_pos = np.where(indices == target[i])[0][0]
#             arc_index_list[0].append(src_pos)
#             arc_index_list[1].append(target_pos)
#             arc_costs_list.append(distance[i])
    
#     arc_index = np.array(arc_index_list)
#     arc_costs = np.array(arc_costs_list)
#     return CVRP(nodes, arc_index, vehicle_capacity, arc_costs, nb_vehicles)

#TW-updated generate_restricted_cvrp_instances_Munich function
def generate_restricted_cvrp_instances_Munich(
    path_nodes: str,
    path_arcs: str,
    vehicle_capacity=30,
    nb_vehicles=5,
    nb_clients=20,
    # ---- TW flags ----
    use_time_windows=False,
    tw_horizon=1000.0,
    tw_width_min=150.0,
    tw_width_max=300.0,
    service_time=0.0,
    tw_seed=0,
):
    """
    Create a restricted sub-instance from Munich data (CVRP/CVRPTW).

    Important: This function re-indexes selected nodes into a contiguous 0..nb_clients space,
    and filters arcs accordingly. If use_time_windows=True, assigns TW fields to nodes.

    Returns
    -------
    CVRP (actually CVRPTW-ready CVRP object with TW fields in nodes)
    """
    df_nodes = pd.read_csv(path_nodes)

    # Original node table columns
    indices_all = df_nodes["Index"].to_numpy()
    x_coords = df_nodes["X"].to_numpy()
    y_coords = df_nodes["Y"].to_numpy()
    demands_raw = df_nodes["Demand/Capacity"].to_numpy()

    # Keep your original behavior: depot row demand stays 0, others random 1..9
    demands_all = np.array(
        [np.random.randint(1, 10) * (demands_raw[i] != 0) for i in range(len(indices_all))],
        dtype=float,
    )

    # Pick a subset: keep original id 0 as depot-candidate, plus nb_clients customers
    random_indices = np.random.permutation(np.arange(1, len(demands_all)))  # exclude 0
    selected_original_ids = np.concatenate(([0], random_indices[:nb_clients]))

    # Build nodes with NEW contiguous IDs 0..nb_clients
    nodes = []
    for new_id, orig_id in enumerate(selected_original_ids):
        nodes.append(
            CVRP_node(
                node_id=int(new_id),
                demand=float(demands_all[int(orig_id)]),
                x=float(x_coords[int(orig_id)]),
                y=float(y_coords[int(orig_id)]),
            )
        )

    # Map original node ids -> new contiguous ids
    orig_to_new = {int(orig): int(new) for new, orig in enumerate(selected_original_ids)}

    # Load arcs (source,target,distance)
    df_arcs = pd.read_csv(path_arcs, header=None, names=["source", "target", "distance"])
    source = df_arcs["source"].to_numpy()
    target = df_arcs["target"].to_numpy()
    distance = df_arcs["distance"].to_numpy()

    arc_index_list = [[], []]
    arc_costs_list = []

    for i, src in enumerate(source):
        dst = target[i]
        if int(src) == int(dst):
            continue
        if int(src) in orig_to_new and int(dst) in orig_to_new:
            arc_index_list[0].append(orig_to_new[int(src)])
            arc_index_list[1].append(orig_to_new[int(dst)])
            arc_costs_list.append(float(distance[i]))

    arc_index = np.array(arc_index_list, dtype=int)
    arc_costs = np.array(arc_costs_list, dtype=float)

    inst = CVRP(nodes, arc_index, vehicle_capacity, arc_costs, nb_vehicles)

    # Assign TW fields if requested
    if use_time_windows:
        assign_time_windows_simple(
            inst.nodes,
            horizon=tw_horizon,
            width_min=tw_width_min,
            width_max=tw_width_max,
            service_time=service_time,
            seed=tw_seed,
        )

    return inst





# def save_sample(instance: CVRP, path: str):
#     # Solve instance to get solution
#     demands = np.array([node.demand for node in instance.nodes])
#     model, x, _ = cvrp(demands,
#                    instance.arc_index,
#                    instance.arc_costs,
#                    instance.nb_vehicles,
#                    instance.vehicle_capacity)
#     model.setParam("OutputFlag", 1)  ##replace 0 by 1 to see the outputflag
#     model.setParam("TimeLimit", 10)
#     model.optimize()


#     if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.SUBOPTIMAL or model.SolCount > 0:
#         print("Optimal objective value:", model.objVal)
#         # for k, var in x.items():
#         #     if var.X > 0.5:  # chosen arcs
#         #         print(f"Arc {k} is used, cost {var.Obj}")
#         sol = sol_vals(x)
#     elif model.status == gp.GRB.INFEASIBLE:
#         model.computeIIS()
#         model.write("model.ilp") 
#         sol = None
#         print("No feasible solution found, status:", model.status)


#     sample = {
#         "instance": instance.to_dict(),
#         "solution": sol,
#         "runtime": model.Runtime,
#         "opt_gap": model.MIPGap,
#         "opt_status": model.Status,
#     }
#     with gzip.open(path, "wb") as f:
#         pkl.dump(sample, f)

#TW-updated save_sample function
def save_sample(instance: CVRP, path: str, cfg: DictConfig):

    demands = np.array([n.demand for n in instance.nodes], dtype=float)

    if cfg.problem.type == "cvrp":
        model, x, _ = cvrp(
            demands,
            instance.arc_index,
            instance.arc_costs,
            cfg.sampling.nb_vehicles,
            cfg.sampling.vehicle_capacity,
        )

    elif cfg.problem.type == "cvrptw":
        ready = np.array([n.ready_time for n in instance.nodes])
        due = np.array([n.due_time for n in instance.nodes])
        service = np.array([n.service_time for n in instance.nodes])

        model, x, _, _ = cvrptw(
            demands,
            instance.arc_index,
            instance.arc_costs,
            cfg.sampling.nb_vehicles,
            cfg.sampling.vehicle_capacity,
            ready,
            due,
            service,
        )

    else:
        raise ValueError(f"Unknown problem type: {cfg.problem.type}")

    model.setParam("OutputFlag", cfg.sampling.outputflag)
    model.setParam("TimeLimit", cfg.sampling.timelimit)
    model.optimize()

    sol = sol_vals(x) if model.SolCount > 0 else None

    sample = {
        "instance": instance.to_dict(),
        "solution": sol,
        "runtime": model.Runtime,
        "opt_gap": getattr(model, "MIPGap", None),
        "opt_status": model.Status,
    }

    with gzip.open(path, "wb") as f:
        pkl.dump(sample, f)



def test_Clark_heuristic(instance):
    # demands = np.array([node.demand for node in instance.nodes])

    sol_dict = Clark_Wright_heuristic(instance.demands,
                                    instance.arc_index,
                                    instance.arc_costs,
                                    instance.nb_vehicles,
                                    instance.vehicle_capacity)
    
    cost_dict = {(int(src), int(dst)): instance.arc_costs[k]
             for k, (src, dst) in enumerate(zip(instance.arc_index[0], instance.arc_index[1]))}
    total_cost = 0
    for arc, usage in sol_dict.items():
        total_cost += cost_dict[arc] * usage

    print("Final solution cost:", total_cost)

def solve_HGS_VRP(instance, path):

    index_node = 0

    m = HGS_Model()
    m.add_vehicle_type(capacity=instance.vehicle_capacity, num_available=instance.nb_vehicles)
    total_nodes_dict = {}

    depot_node = instance.depot
    depot_node_id = depot_node.node_id
    depot = m.add_depot(x=depot_node.x, y=depot_node.y, name="f{depot_node.node_id}")

    index_node += 1
    total_nodes_dict[depot_node.node_id] = depot
    for node in instance.clients: 
        total_nodes_dict[node.node_id] = m.add_client(x=node.x, y=node.y, delivery=int(node.demand), name = "f{node.node_id}")
        
    for i , arc in enumerate(instance.arc_list):
        m.add_edge(total_nodes_dict[arc[0]], total_nodes_dict[arc[1]],
                    distance=instance.arc_costs[i])
        # print("(",arc[0], ", ", arc[1], ") : ", instance.arc_costs[i] )
    res = m.solve(stop=HGS_MaxRuntime(3)) 
    print(res)
    sol_dict = {(int(src), int(dst)): 0
             for (src, dst) in (zip(instance.arc_index[0], instance.arc_index[1]))}
    for route in res.best.routes():
        visits = route.visits()

        # now build sol_dict using original IDs
        sol_dict[(depot_node_id, visits[0])] = 1
        sol_dict[(visits[-1], depot_node_id)] = 1
        for j in range(1, len(visits)):
            sol_dict[(visits[j-1], visits[j])] = 1


    sample = {
        "instance": instance.to_dict(),
        "solution": sol_dict,
        "runtime": res.runtime,
        "opt_gap": None,
        "opt_status": "heuristic HGS",
    }
    with gzip.open(path, "wb") as f:
        pkl.dump(sample, f)



##Test clusters

def solve_HGS_VRP_test_cluster(instance, path):

    index_node = 0

    m = HGS_Model()
    m.add_vehicle_type(capacity=instance.vehicle_capacity, num_available=instance.nb_vehicles)
    total_nodes_dict = {}

    depot_node = instance.depot
    depot_node_id = depot_node.node_id
    depot = m.add_depot(x=depot_node.x, y=depot_node.y, name="f{depot_node.node_id}")

    index_node += 1
    total_nodes_dict[depot_node.node_id] = depot
    for node in instance.clients: 
        total_nodes_dict[node.node_id] = m.add_client(x=node.x, y=node.y, delivery=int(node.demand), name = "f{node.node_id}")
        
    for i , arc in enumerate(instance.arc_list):
        m.add_edge(total_nodes_dict[arc[0]], total_nodes_dict[arc[1]],
                    distance=instance.arc_costs[i])
        # print("(",arc[0], ", ", arc[1], ") : ", instance.arc_costs[i] )
    res = m.solve(stop=HGS_MaxRuntime(3)) 
    print(res)
    sol_dict = {(int(src), int(dst)): 0
             for (src, dst) in (zip(instance.arc_index[0], instance.arc_index[1]))}
    for route in res.best.routes():
        visits = route.visits()

        # now build sol_dict using original IDs

        for i in range(len(visits)-1):
            for j in range(i+1, len(visits)):
                sol_dict[(visits[i], visits[j])] = 1
                sol_dict[(visits[j], visits[i])] = 1

    sample = {
        "instance": instance.to_dict(),
        "solution": sol_dict,
        "runtime": res.runtime,
        "opt_gap": None,
        "opt_status": "heuristic HGS",
    }
    with gzip.open(path, "wb") as f:
        pkl.dump(sample, f)

def generate_benchmark_plot(solution_dir):
    num_arcs_pred_data, num_arcs_enriched_data = get_num_edges_by_method(solution_dir)
    runtime_data = get_runtimes_by_method(solution_dir)
    solver_runtime_data = get_solver_runtimes_by_method(solution_dir)
    mip_gap_data = get_mip_gap_by_method(solution_dir)
    opt_gap_data = get_optgaps_by_method(solution_dir)
    objval_data = get_objval_by_method(solution_dir)
    num_arcs_used_data = get_num_arcs_used_by_method(solution_dir)

    print(num_arcs_used_data)


    x = range(len(next(iter(opt_gap_data.values())))) 

    num_methods = len(opt_gap_data)
    fig, axes = plt.subplots(num_methods, 1, figsize=(10, 6), sharex=True)

    if num_methods == 1:
        axes = [axes]

    for ax, (method, values) in zip(axes, opt_gap_data.items()):
        ax.plot(x, values[:len(next(iter(opt_gap_data.values())))], marker='o', linestyle='-', label=method)
        ax.set_title(f"Method: {method}", fontsize=10)
        ax.set_ylabel("Opt_gap_value %")
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=8)

    # Final touches
    axes[-1].set_xlabel("Instance index")
    plt.tight_layout()
    plt.show()


def parse_cvrp_literatur_instances(path_instance):
    nb_vehicles = 2*int(re.search(r'k(\d+)', path_instance).group(1))
    with open(path_instance, "r") as f :
        lines = f.readlines()

    coords = {}
    demands = {}
    vehicle_capacity = None
    mode = None

    for line in lines : 
        line = line.strip()

        if line.startswith("CAPACITY"):
            vehicle_capacity = int(line.split(":")[1])

        if line.startswith("NODE_COORD_SECTION"):
            mode="coords"
            continue
        if line.startswith("DEMAND_SECTION"):
            mode="demands"
            continue
        if line.startswith("DEPOT_SECTION"): 
            mode = "depot"
            continue
        if line.startswith("EOF"):
            break

        if mode=="coords" and line:
            parts = line.split()
            idx = int(parts[0])-1
            x = float(parts[1])
            y = float(parts[2])
            coords[idx] = (x, y)

        if mode=="demands" and line:
            parts = line.split()
            idx = int(parts[0])-1
            demands[idx] = int(parts[1])


        

    nodes = []

    for idx in sorted(coords.keys()):
        x, y = coords[idx]
        d = demands[idx]
        nodes.append(CVRP_node(idx, d, x, y))

    arc_index_list = [[],[]]
    arc_cost_list = []

    node_ids = sorted(coords.keys())

    for i in node_ids:
        x1, y1 = coords[i]
        for j in node_ids:
            if i != j:
                x2, y2 = coords[j]

                dist = round(math.sqrt((x1-x2)**2 + (y1-y2)**2))
                arc_index_list[0].append(i)
                arc_index_list[1].append(j)
                arc_cost_list.append(dist)

    arc_index = np.array(arc_index_list)
    arc_costs = np.array(arc_cost_list)
    return CVRP(nodes, arc_index, vehicle_capacity, arc_costs, nb_vehicles)

def parse_solution_file(path, depot=0):
    arcs = {}   # dictionary of arcs (u, v) → 1

    with open(path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Stop at cost line
        if line.startswith("Cost"):
            break

        # Parse route lines
        if line.startswith("Route"):
            # Example: "Route #1: 31 46 35"
            parts = line.split(":")[1].strip().split()
            route = [int(x) for x in parts]

            # Add depot → first
            arcs[(depot, route[0])] = 1

            # Add internal arcs
            for u, v in zip(route[:-1], route[1:]):
                arcs[(u, v)] = 1

            # Add last → depot
            arcs[(route[-1], depot)] = 1

    return arcs




# --- Main: generate a few samples ---

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="configs/training", config_name="config")
def main(cfg: DictConfig):
    out_dir = OmegaConf.select(cfg, "paths.out_dir") or "data/samples_out"
    os.makedirs(out_dir, exist_ok=True)

    checkpoint_path = (
        OmegaConf.select(cfg, "debug.checkpoint_path")
        or "trained_models_X_instances_mix_Munich_1000_100/"
           "model_gcnn_features_graph_raw_prediction_task_binary_classification_"
           "normalization_standard_hidden_layer_dim_20_num_conv_layers_6_num_dense_layers_2/"
           "cross_val/fold_0/checkpoint.pth.tar"
    )

    problem_type = (OmegaConf.select(cfg, "problem.type") or "cvrp").lower()

    # NEW: how many samples?
    num_samples = int(OmegaConf.select(cfg, "num_samples") or 1)


    for k in range(num_samples):
        out_path = os.path.join(out_dir, f"sample_{k}.pkl.gz")

        # deterministic per-sample RNG (so CVRP and CVRPTW runs match with same seed)
        base_seed = int(OmegaConf.select(cfg, "seed") or 0)
        np.random.seed(base_seed + k)

        # -----------------------------
        # Generate instance
        # -----------------------------
        instance = generate_cvrp_instance(
            num_nodes=cfg.sampling.nb_clients + 1,  # depot + clients
            vehicle_capacity=cfg.sampling.vehicle_capacity,
            nb_vehicles=cfg.sampling.nb_vehicles,
        )

        # add TW only if cvrptw
        if problem_type == "cvrptw":
            tw_seed = int(cfg.problem.time_windows.seed) + k
            assign_time_windows_simple(
                instance.nodes,
                horizon=cfg.problem.time_windows.horizon,
                width_min=cfg.problem.time_windows.width_min,
                width_max=cfg.problem.time_windows.width_max,
                service_time=cfg.problem.time_windows.service_time,
                seed=tw_seed,
            )
        elif problem_type != "cvrp":
            raise ValueError(f"Unknown problem type: {cfg.problem.type}")

        save_sample(instance=instance, path=out_path, cfg=cfg)
        print(f"[OK] Sample saved to: {out_path}")

    # ----- plotting part unchanged -----
    if not os.path.exists(checkpoint_path):
        print(f"[WARN] Checkpoint not found, skipping plots:\n  {checkpoint_path}")
        return

    chkpnt = torch.load(checkpoint_path, map_location="cpu")
    perf = chkpnt["exp_dict"]["performance"]

    ConfusionMatrixDisplay(confusion_matrix=perf["confusion_matrix"][-1]).plot(cmap="Blues")
    plt.title("Final Confusion Matrix")
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(perf["train_loss"], label="Training Loss")
    axes[0].plot(perf["validation_loss"], label="Validation Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(perf["validation_accuracy"], label="Validation Accuracy", color="green")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(perf["validation_recall"], label="Recall", color="orange")
    axes[2].plot(perf["validation_precision"], label="Precision", color="blue")
    axes[2].plot(perf["validation_fscore"], label="F-score", color="red")
    axes[2].set_title("Validation Metrics")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()



#     # np.random.seed(3) #first 42, now 4, now 3
#     # os.makedirs("data/samples_Munich_100_test_cluster", exist_ok=True)
#     # for k in range(1):  
#     #     # rd_num_nodes = np.random.randint(30, 50)
#     #     # print("rd_num_nodes = ", rd_num_nodes)
#     #     # inst = generate_cvrp_instance(rd_num_nodes, 30, rd_num_nodes)
#     #     Munich_inst = generate_restricted_cvrp_instances_Munich(
#     #     "interlog_gen-master/resources/instances/100_0_1/all_w_geom.csv",
#     #     "interlog_gen-master/resources/instances/100_0_1/dm_drive.csv", nb_clients= 20)
#     #     # save_sample(Munich_inst, f"data/samples_Munich/sample_20_0_1_Munich{k}.pkl.gz")
#     #     # test_Clark_heuristic(inst)
#     #     # solve_HGS_VRP_test_cluster(Munich_inst, f"data/samples_Munich_100_test_cluster/sample_100_0_1_Munich{k}.pkl.gz")
#     #     connections = [True]*len(Munich_inst.arc_costs)
#     #     arc_cost_neg = [-np.random.randint(30, 50) for i in range(len(Munich_inst.arc_costs))]
#     #     cvrp_via_VRP_Easy(Munich_inst.demands,
#     #                Munich_inst.arc_index,
#     #                Munich_inst.arc_costs,
#     #                Munich_inst.nb_vehicles,
#     #                Munich_inst.vehicle_capacity, connections)
#     chkpnt = torch.load("trained_models_X_instances_mix_Munich_1000_100/model_gcnn_features_graph_raw_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_20_num_conv_layers_6_num_dense_layers_2/cross_val/fold_0/checkpoint.pth.tar", map_location="cpu")    

#     perf = chkpnt["exp_dict"]["performance"]

#     ConfusionMatrixDisplay(confusion_matrix=perf["confusion_matrix"][-1]).plot(cmap="Blues")
#     plt.title("Final Confusion Matrix")
#     plt.show()

#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))

#     # --- Loss ---
#     axes[0].plot(perf["train_loss"], label="Training Loss")
#     axes[0].plot(perf["validation_loss"], label="Validation Loss")
#     axes[0].set_title("Loss")
#     axes[0].set_xlabel("Epoch")
#     axes[0].set_ylabel("Loss")
#     axes[0].legend()
#     axes[0].grid(True)

#     # --- Accuracy ---
#     axes[1].plot(perf["validation_accuracy"], label="Validation Accuracy", color="green")
#     axes[1].set_title("Accuracy")
#     axes[1].set_xlabel("Epoch")
#     axes[1].set_ylabel("Accuracy")
#     axes[1].legend()
#     axes[1].grid(True)

#     # --- Recall / Precision / F-score ---
#     axes[2].plot(perf["validation_recall"], label="Recall", color="orange")
#     axes[2].plot(perf["validation_precision"], label="Precision", color="blue")
#     axes[2].plot(perf["validation_fscore"], label="F-score", color="red")
#     axes[2].set_title("Validation Metrics")
#     axes[2].set_xlabel("Epoch")
#     axes[2].set_ylabel("Score")
#     axes[2].legend()
#     axes[2].grid(True)

#     plt.tight_layout()
#     plt.show()


    # # files = os.listdir("trained_models_Munich_100/model_gcnn_features_graph_raw_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_20_num_conv_layers_6_num_dense_layers_2")
    # # print("Files in directory:", files)

    # generate_benchmark_plot("benchmarking")

    # print("Samples generated in data/samples/")

    # import os
    # import ctypes

    # # Add CPLEX DLL directory explicitly
    # os.add_dll_directory(r"C:\Users\bapti\IBM_CPLEX\cplex\bin\x64_win64")

    # # Now load your DLL
    # lib = ctypes.CDLL(r"C:\Users\bapti\Documents\Neuer Ordner\Reduce-then-Optimize-for-FCTP\bapcod-shared.dll")



    # import VRPSolverEasy as vrpse
    # import math

    # def compute_euclidean_distance(x_i, x_j, y_i, y_j):
    #     """compute the euclidean distance between 2 points from graph"""
    #     return round(math.sqrt((x_i - x_j)**2 +
    #                         (y_i - y_j)**2), 3)

    # # Data
    # cost_per_distance = 10
    # begin_time = 0
    # end_time = 5000
    # nb_point = 7

    # # Map with names and coordinates
    # coordinates = {"Wisconsin, USA": (44.50, -89.50),  # depot
    #             "West Virginia, USA": (39.000000, -80.500000),
    #             "Vermont, USA": (44.000000, -72.699997),
    #             "Texas, the USA": (31.000000, -100.000000),
    #             "South Dakota, the US": (44.500000, -100.000000),
    #             "Rhode Island, the US": (41.742325, -71.742332),
    #             "Oregon, the US": (44.000000, -120.500000)
    #             }

    # # Demands of points
    # demands = [0, 500, 300, 600, 658, 741, 436]

    # # Initialisation
    # model = vrpse.Model()

    # # Add vehicle type
    # model.add_vehicle_type(
    #     id=1,
    #     start_point_id=0,
    #     end_point_id=0,
    #     name="VEH1",
    #     capacity=1100,
    #     max_number=6,
    #     var_cost_dist=cost_per_distance,
    #     tw_end=5000)

    # # Add depot
    # model.add_depot(id=0, name="D1", tw_begin=0, tw_end=5000)

    # coordinates_keys = list(coordinates.keys())
    # # Add customers
    # for i in range(1, nb_point):
    #     model.add_customer(
    #         id=i,
    #         name=coordinates_keys[i],
    #         demand=demands[i],
    #         tw_begin=begin_time,
    #         tw_end=end_time)

    # # Add links
    # coordinates_values = list(coordinates.values())
    # for i in range(0, 7):
    #     for j in range(i + 1, 7):
    #         dist = compute_euclidean_distance(coordinates_values[i][0],
    #                                         coordinates_values[j][0],
    #                                         coordinates_values[i][1],
    #                                         coordinates_values[j][1])
    #         model.add_link(
    #             start_point_id=i,
    #             end_point_id=j,
    #             distance=dist,
    #             time=dist)

    # # solve model
    # model.solve()
    # model.export()

    # if model.solution.is_defined():
    #     print(model.solution)

    # folder_path = "CVRPLIB-main\X\Instances"
    # solution_folder_path = "CVRPLIB-main\X\Solutions"
    # os.makedirs("data/X_instances", exist_ok=True)
    # cvrp_instances = []

    # for filename in os.listdir(folder_path):
    #     instance_path = os.path.join(folder_path, filename)
    #     solution_filename = filename.replace(".vrp", ".sol")


    #     solution_path = os.path.join(solution_folder_path, solution_filename)

    #     cvrp_instance = parse_cvrp_literatur_instances(instance_path)
    #     solution_cvrp_instance = parse_solution_file(solution_path)


    #     arc_indicator = {} # All arcs in the instance
    #     sources = cvrp_instance.arc_index[0]
    #     targets = cvrp_instance.arc_index[1]

    #     for u, v in zip(sources, targets):
    #         if (u, v) in solution_cvrp_instance:
    #             arc_indicator[(u, v)] = 1 
    #         else:
    #             arc_indicator[(u, v)] = 0

    #     sample = {
    #         "instance": cvrp_instance.to_dict(),
    #         "solution": arc_indicator,
    #         "runtime": 0,
    #         "opt_gap": None,
    #         "opt_status": "best_opt",
    #     }
    #     save_filename = filename.replace(".vrp", "")
    #     with gzip.open(f"data/X_instances/" +save_filename +".pkl.gz", "wb") as f:
    #         pkl.dump(sample, f)



        # print(f"Number of nodes: {len(cvrp_instance.nodes)}")
        # print(f"Vehicle capacity: {cvrp_instance.vehicle_capacity}")
        # print(f"Number of vehicles: {cvrp_instance.nb_vehicles}")

        # Depot is the node with demand 0
        # depot_nodes = [n.node_id for n in cvrp_instance.nodes if n.demand == 0]
        # print(f"Depot node(s): {depot_nodes}")

        # Print first few nodes to check parsing
        # print("\nFirst 5 nodes:")
        # for n in cvrp_instance.nodes[:5]:
        #     print(f"  Node {n.node_id}: demand={n.demand}, x={n.x}, y={n.y}")

        # # Arc info
        # print(f"\nNumber of arcs: {cvrp_instance.arc_index.shape[1]}")
        # print("First 5 arcs:")
        # for i in range(5):
        #     src = cvrp_instance.arc_index[0][i]
        #     tgt = cvrp_instance.arc_index[1][i]
        #     cost = cvrp_instance.arc_costs[i]
        #     print(f"  {src} -> {tgt} : cost={cost}")

        # print("==============================\n")

"""Python wrapper for Julia implementations of FCTP meta-heuristics. """

import os

import numpy as np

from core.utils.utils import get_random_string
# from core.utils.postprocessing import matrix_to_dict


# class TabuSearchJuliaEnv:
#     """Wrapper for Julia implementation of Tabu Search."""

#     def __init__(self):
#         from julia.api import Julia

#         jl = Julia(compiled_modules=False)
#         jl.eval('include("core/fctp_heuristics_julia/sun_ts.jl")')
#         self.ts = jl.eval("TabuSearch.ts_wrapper")
#         self.compile_run()

#     def compile_run(self):
#         """Perform compiliation run."""
#         supply = np.array([4, 2, 3])
#         demand = np.array([4, 3, 2])
#         var_costs = np.array([[3.0, 2.0, 4.0], [5.0, 2.0, 2.0], [3.0, 4.0, 3.0]])
#         fix_costs = np.array(
#             [[40.0, 35.0, 30.0], [25.0, 45.0, 30.0], [35.0, 35.0, 40.0]]
#         )
#         edge_mask = [[True, False, True], [True, True, False], [False, True, True]]
#         bfs = {(1, 1): 4, (2, 1): 0, (2, 2): 2, (3, 2): 1, (3, 3): 2}
#         test_log_file = f"{get_random_string(10)}_test.txt"
#         config = {
#             "tabu_in_range": (7, 10),
#             "tabu_out_range": (2, 4),
#             "beta": 0.5,
#             "gamma": 0.5,
#             "L": 3,
#             "seed": 0,
#             "log_file": test_log_file,
#         }
#         self.ts(supply, demand, var_costs, fix_costs, edge_mask, bfs, config)
#         try:
#             os.remove(test_log_file)
#         except OSError:
#             pass

#     def run(self, instance, bfs, config, edge_mask=None):
#         """Run Tabu Search to solve FCTP.

#         Parameters
#         ----------
#         instance: FCTP
#             Instance to solve.
#         bfs: dict
#             Basic feasible solution (start solution).
#         config: dict
#             Tabu Search configuration.
#         edge_mask: 2D np.array, optional
#             Boolean edge mask indicating relevant edges.

#         Returns
#         -------
#         dict:
#             Solution dictionary.
#         sol_val: float
#             Objective function value.
#         runtime: float
#             Runtime.

#         """
#         if edge_mask is None:
#             edge_mask = np.full(instance.var_costs.shape, True, dtype=bool)
#         bfs = {(i + 1, j + 1): int(v) for (i, j), v in bfs.items()}
#         sol, sol_val, runtime = self.ts(
#             instance.supply.astype(int),
#             instance.demand.astype(int),
#             instance.var_costs.astype("float64"),
#             instance.fix_costs.astype("float64"),
#             edge_mask,
#             bfs,
#             config,
#         )
#         return matrix_to_dict(sol, False), sol_val, runtime


# class EvolutionaryAlgorithmJuliaEnv:
#     """Wrapper for Julia implementation of Evolutionary Algorithm."""

#     def __init__(self):
#         from julia.api import Julia

#         # Load julia function
#         jl = Julia(compiled_modules=False)
#         jl.eval('include("core/fctp_heuristics_julia/eckert_ea.jl")')
#         self.ea = jl.eval("EvolutionaryAlgorithm.ea_wrapper")
#         self.compile_run()

#     def compile_run(self):
#         """Perform compiliation run."""
#         supply = np.array([4, 2, 3])
#         demand = np.array([4, 3, 2])
#         var_costs = np.array([[3.0, 2.0, 4.0], [5.0, 2.0, 2.0], [3.0, 4.0, 3.0]])
#         fix_costs = np.array(
#             [[40.0, 35.0, 30.0], [25.0, 45.0, 30.0], [35.0, 35.0, 40.0]]
#         )
#         edge_mask = [[False, True, True], [True, True, False], [True, True, True]]
#         for mutation_operator in ["eicr", "nlo"]:
#             test_log_file = f"{get_random_string(10)}_test.txt"
#             config = {
#                 "pop_size": 3,
#                 "max_unique_sols": 10,
#                 "patience": 100,
#                 "mutation_operator": mutation_operator,
#                 "seed": 0,
#                 "log_file": test_log_file,
#             }
#             self.ea(supply, demand, var_costs, fix_costs, edge_mask, config)
#             try:
#                 os.remove(test_log_file)
#             except OSError:
#                 pass

#     def run(self, instance, config, edge_mask=None):
#         """Run EA to solve FCTP.

#         Parameters
#         ----------
#         instance: FCTP
#             FCTP instance to solve.
#         config: dict
#             EA configuration.
#         edge_mask: 2D np.array, optional
#             Boolean edge mask indicating relevant edges.

#         Returns
#         -------
#         dict:
#             Solution dictionary.
#         sol_val: float
#             Objective function value.
#         runtime: float
#             Runtime.

#         """
#         if edge_mask is None:
#             edge_mask = np.full(instance.var_costs.shape, True, dtype=bool)
#         sol, sol_val, runtime = self.ea(
#             instance.supply,
#             instance.demand,
#             instance.var_costs.astype("float64"),
#             instance.fix_costs.astype("float64"),
#             edge_mask,
#             config,
#         )
#         return matrix_to_dict(sol, False), sol_val, runtime


##Changes

class Frank_Wolfe_regularisation_env:
    def __init__(self):

        from juliacall import Main as jl

        jl.seval("""
            import Pkg
            Pkg.activate("C:/Users/bapti/Documents/Neuer Ordner/Reduce-then-Optimize-for-FCTP/.venv/julia_env")
            """)
        jl.include("core/fctp_heuristics_julia/Frank_Wolfe_regularisation.jl")
        self.FW = jl.Frank_Wolfe_regularisation.compute_argmax_relaxed_regularized_CVRP_FW
        self.compile_run()

    def compile_run(self):
        """Perform compilation run"""

        from juliacall import convert as jlconvert, Main as jl

        demands = np.array([0, 2, 3, 1])
        nb_vehicles = 1
        capacity_vehicles = 6
        arcs_list = [(i + 1, j+1) for i in range(len(demands)) for j in range(len(demands)) if i != j]

        # y_values = {(1,2) : 1.0, (2,3) : 1.0, (3,4) : 1.0, (4,1) : 1.0}

        # f_values = {(1,2) : 0.0, (2,3) : 2.0, (3,4) : 5.0, (4,1) : 6.0}

        arc_costs = {(i, j) : np.random.rand() for (i,j) in arcs_list}
        
        coef_lambda = 0.5

        max_iteration = 50

        print("Running compilation run...")

        y_arc_sol, f_arc_sol, sol_val = self.FW(
            demands,
            arcs_list,
            arc_costs,
            nb_vehicles,
            capacity_vehicles,
            coef_lambda,
            max_iteration
        )
        print(y_arc_sol)
        print(f_arc_sol)
        print(sol_val)
        
    def run(self, instance, arc_costs, coef_lambda, max_iteration):

        arcs_list = [(int(src) + 1, int(dst) + 1) for src, dst in zip(instance.arc_index[0], instance.arc_index[1])]
        cost_dict = {(int(src) + 1, int(dst) + 1): arc_costs[k]
             for k, (src, dst) in enumerate(zip(instance.arc_index[0], instance.arc_index[1]))}

        y_sol, f_sol, sol_val = self.FW(instance.demands, arcs_list, cost_dict,
                instance.nb_vehicles, instance.vehicle_capacity,
                coef_lambda, max_iteration)
        
        print(sol_val)
        return y_sol, f_sol, sol_val
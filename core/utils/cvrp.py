# """Classes and functionalities to represent FCTP instances."""
##Changes
"""Classes and functionalities to represent CVRP instances."""
##End changes

import numpy as np


# class FCTP:
#     """Class representing FCTP instance.

#     Attributes
#     ----------
#     supply: np.array
#         Supply vector.
#     demand: np.array
#         Demand vector.
#     var_costs: np.array
#         Variable cost matrix.
#     fix_costs: np.array
#         Fixed-cost matrix.
#     supplier_locations: np.array, optional
#         Supplier locations (m x 2).
#     customer_locations: np.array, optional
#         Customer locations (m x 2).

#     """

#     def __init__(
#         self,
#         supply,
#         demand,
#         var_costs,
#         fix_costs,
#         supplier_locations=None,
#         customer_locations=None,
#     ):
#         self.supply = supply
#         self.demand = demand
#         self.var_costs = var_costs
#         self.fix_costs = fix_costs
#         self.supplier_locations = supplier_locations
#         self.customer_locations = customer_locations

#     @property
#     def m(self):
#         """Number of supply nodes."""
#         return len(self.supply)

#     @property
#     def n(self):
#         """Number of demand nodes."""
#         return len(self.demand)

#     def to_dict(self):
#         attributes = vars(self)
#         attributes["instance_type"] = "fctp"
#         return attributes

#     def eval_sol_dict(self, solution):
#         """Compute total costs for solution provided as dict.

#         Parameters
#         ----------
#         solution: dict
#             Solution dictionary containing flows from supplier i to customer j.

#         Returns
#         -------
#         float or int
#             Total costs incurred by solution.

#         """
#         v_costs = sum([v * self.var_costs[i, j] for (i, j), v in solution.items()])
#         f_costs = sum([self.fix_costs[i, j] for (i, j), v in solution.items() if v > 0])
#         return v_costs + f_costs

#     def eval_sol_matrix(self, solution):
#         """Compute total costs for solution provided as matrix.

#         Parameters
#         ----------
#         solution: np.array
#             Solution matrix (m x n) containing flows from supplier i to customer j.

#         Returns
#         -------
#         float or int
#             Total costs incurred by solution.

#         """
#         return np.sum(
#             solution * self.var_costs + np.where(solution > 0, 1, 0) * self.fix_costs,
#         )


# class CapacitatedFCTP(FCTP):
#     """Class representing capacitated FCTP instance.

#     Attributes
#     ----------
#     supply: np.array
#         Supply vector.
#     demand: np.array
#         Demand vector.
#     var_costs: np.array
#         Variable cost matrix (m x n).
#     fix_costs: np.array
#         Fixed-cost matrix (m x n).
#     edge_capacities: np.array
#         Edge capacity matrix (m x n).
#     supplier_locations: np.array, optional
#         Supplier locations (m x 2).
#     customer_locations: np.array, optional
#         Customer locations (m x 2).

#     """

#     def __init__(
#         self,
#         supply,
#         demand,
#         var_costs,
#         fix_costs,
#         edge_capacities,
#         supplier_locations=None,
#         customer_locations=None,
#     ):
#         super().__init__(
#             supply, demand, var_costs, fix_costs, supplier_locations, customer_locations
#         )

#         self.edge_capacities = edge_capacities

#     def to_dict(self):
#         attributes = vars(self)
#         attributes["instance_type"] = "c-fctp"
#         return attributes


# class FixedStepFCTP(FCTP):
#     """Class representing capacitated FCTP instance with fixed-step costs.

#     Attributes
#     ----------
#     supply: np.array
#         Supply vector.
#     demand: np.array
#         Demand vector.
#     var_costs: np.array
#         Variable cost matrix (m x n).
#     fix_costs: np.array
#         Fixed-cost matrix (m x n).
#     vehicle_capacities: np.array
#         Vehicle capacity matrix (m x n).
#     supplier_locations: np.array, optional
#         Supplier locations (m x 2).
#     customer_locations: np.array, optional
#         Customer locations (m x 2).

#     """

#     def __init__(
#         self,
#         supply,
#         demand,
#         var_costs,
#         fix_costs,
#         vehicle_capacities,
#         supplier_locations=None,
#         customer_locations=None,
#     ):
#         super().__init__(
#             supply, demand, var_costs, fix_costs, supplier_locations, customer_locations
#         )

#         self.vehicle_capacities = vehicle_capacities

#     def to_dict(self):
#         attributes = vars(self)
#         attributes["instance_type"] = "fs-fctp"
#         return attributes

#     def eval_sol_dict(self, solution):
#         """Compute total costs for solution provided as dict.

#         Parameters
#         ----------
#         solution: dict
#             Solution dictionary containing flows from supplier i to customer j.

#         Returns
#         -------
#         float or int
#             Total costs incurred by solution.

#         """
#         v_costs = sum([v * self.var_costs[i, j] for (i, j), v in solution.items()])
#         f_costs = sum(
#             [
#                 self.fix_costs[i, j] * np.ceil(v / self.vehicle_capacities[i, j])
#                 for (i, j), v in solution.items()
#                 if v > 0
#             ]
#         )
#         return v_costs + f_costs

#     def eval_sol_matrix(self, solution):
#         """Compute total costs for solution provided as matrix.

#         Parameters
#         ----------
#         solution: np.array
#             Solution matrix (m x n) containing flows from supplier i to customer j.

#         Returns
#         -------
#         float or int
#             Total costs incurred by solution.

#         """
#         v_costs = solution * self.var_costs
#         f_costs = np.ceil(solution / self.vehicle_capacities) * self.fix_costs
#         return np.sum(v_costs + f_costs)
    

##Changes

# class CVRP_node:
#     """Class representing node for CVRP instances

#     Attributes
#     ----------

#     node_id: int
#         Node id.
#     demand: float
#         Node demand.
#     x: float
#         Node x-coordinate.
#     y: float
#         Node y-coordinate    

#     """

#     def __init__(
#             self,
#             node_id,
#             demand,
#             x = None,
#             y = None
#     ):
#         self.node_id = node_id
#         self.demand = demand
#         self.x = x
#         self.y = y

#     def to_dict(self):
#         attributes = vars(self)
#         return attributes

#TW-updated cvrp_node class   
class CVRP_node:
    """Class representing a node for CVRP / CVRPTW instances.

    This class is intentionally designed to be backward-compatible:
    - For classical CVRP instances, time window attributes can be left unset.
    - For CVRPTW instances, time window attributes are provided and consumed
      by downstream components (e.g., feature extraction or optimization).

    Attributes
    ----------
    node_id : int
        Unique identifier of the node.
    demand : float
        Demand associated with the node.
    x : float, optional
        X-coordinate of the node.
    y : float, optional
        Y-coordinate of the node.
    ready_time : float, optional
        Earliest time at which service at the node can start (a_i).
        If None, the node is treated as having no time window.
    due_time : float, optional
        Latest time at which service at the node can start (b_i).
        If None, the node is treated as having no time window.
    service_time : float
        Service duration at the node (s_i). Defaults to 0.0.
    """

    def __init__(
        self,
        node_id,
        demand,
        x=None,
        y=None,
        ready_time=None,
        due_time=None,
        service_time=0.0,
    ):
        self.node_id = node_id
        self.demand = demand
        self.x = x
        self.y = y

        # Time window attributes (optional)
        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = service_time

    def to_dict(self):
        """Convert node attributes to dictionary representation.

        Returns
        -------
        dict
            Dictionary containing all node attributes. Optional time window
            fields are included only if present.
        """
        return vars(self)

    

# class CVRP:
#     """Class representing CVRP instance.

#     Attributes
#     ----------
#         nodes : list of CVRP_nodes
#             List of nodes including depot and clients.
#         arc_index : 2D np.array
#             List of arcs with source and destination
#         vehicle_capacity: float
#             Maximum capacity of ecah vehicle.
#         arc_costs : np array
#             Variable cost matrix.
#         nb_vehicles : int
#             Number of vehicles.
#         arc_lookup : dict
#             Retrive the index of an arc with the pair (i,j)

#     """

#     def __init__(
#         self,
#         nodes,
#         arc_index,
#         vehicle_capacity,
#         arc_costs,
#         nb_vehicles
#     ):
#         self.nodes = nodes
#         self.arc_index = arc_index
#         self.vehicle_capacity = vehicle_capacity
#         self.arc_costs = arc_costs
#         self.nb_vehicles = nb_vehicles
#         self.arc_lookup = {
#             (int(i), int(j)): idx
#             for idx, (i, j) in enumerate(zip(arc_index[0], arc_index[1]))
#         }


#     @property
#     def n(self):
#         """Number of clients nodes."""
#         return len(self.nodes[1:])
    
#     @property
#     def m(self):
#         """Number of arcs."""
#         return len(self.arc_index[0])
    
#     @property
#     def depot(self):
#         """Depot node."""
#         for node in self.nodes:
#             if node.demand == 0:
#                 return node
#         raise ValueError("Depot node not found (no node with demand == 0)")
    
#     @property
#     def clients(self):
#         """Client nodes list."""
#         return [node for node in self.nodes if node.demand > 0]
    
#     @property
#     def demands(self):
#         """Demand values."""
#         return [node.demand for node in self.nodes]
#     @property
#     def arc_list(self):
#         """List of arcs"""
#         return [(int(src), int(dst)) for src, dst in zip(self.arc_index[0], self.arc_index[1])]
    

#     def to_dict(self):

#         return {
#             "nodes" : [node.to_dict() for node in self.nodes],
#             "arc_index" : self.arc_index,
#             # "arc_lookup" : self.arc_lookup,
#             "vehicle_capacity" : self.vehicle_capacity,
#             "arc_costs" : self.arc_costs,
#             "nb_vehicles" : self.nb_vehicles,
#             "instance_type" :  "cvrp"
#         }

#     def eval_sol_dict(self, solution):
#         """Compute total costs for solution provided as dict.

#         Parameters
#         ----------
#         solution: dict
#             Solution dictionary containing booleans for the activation of arcs.

#         Returns
#         -------
#         float or int
#             Total costs incurred by solution.

#         """

#         return sum(
#             v * self.arc_costs[self.arc_lookup[(i, j)]]
#             for (i, j), v in solution.items()
#             if (i, j) in self.arc_lookup
#         )

#     def eval_sol_list(self, solution):
#         """Compute total costs for solution provided as matrix.

#         Parameters
#         ----------
#         solution: np.array
#             Solution list containing booleans for the activation of arc (i,j).

#         Returns
#         -------
#         float or int
#             Total costs incurred by solution.

#         """
#         return np.sum(
#             solution * self.arc_costs,
#         )

# TW-updated cvrp class

class CVRP:
    """Class representing a CVRP / CVRPTW instance.

    This class stores the graph structure (arcs), costs, and vehicle parameters.
    It is compatible with both:
    - CVRP: nodes may not carry time window information
    - CVRPTW: nodes can optionally include ready_time / due_time / service_time

    Attributes
    ----------
    nodes : list of CVRP_node
        List of nodes including depot and clients.
    arc_index : 2D np.array
        Array of arcs with source and destination indices (shape: (2, m)).
    vehicle_capacity : float
        Maximum capacity of each vehicle.
    arc_costs : np.array
        Array of costs for each arc (aligned with arc_index).
    nb_vehicles : int
        Number of vehicles.
    arc_lookup : dict
        Retrieves the index of an arc given a pair (i, j), i.e., (i, j) -> idx.
        Used for O(1) cost lookup when evaluating solutions.
    """

    def __init__(
        self,
        nodes,
        arc_index,
        vehicle_capacity,
        arc_costs,
        nb_vehicles
    ):
        self.nodes = nodes
        self.arc_index = arc_index
        self.vehicle_capacity = vehicle_capacity
        self.arc_costs = arc_costs
        self.nb_vehicles = nb_vehicles

    
        self.arc_lookup = {
            (int(i), int(j)): idx
            for idx, (i, j) in enumerate(zip(arc_index[0], arc_index[1]))
        }

    @property
    def n(self):
        """Number of client nodes (excluding the depot)."""
        return len(self.nodes[1:])

    @property
    def m(self):
        """Number of arcs."""
        return len(self.arc_index[0])

    @property
    def depot(self):
        """Depot node.

        Notes
        -----
        Assumes the depot has demand == 0 (as in the original implementation).
        """
        for node in self.nodes:
            if node.demand == 0:
                return node
        raise ValueError("Depot node not found (no node with demand == 0)")

    @property
    def clients(self):
        """List of client nodes (demand > 0)."""
        return [node for node in self.nodes if node.demand > 0]

    @property
    def demands(self):
        """Demand values for all nodes (including depot)."""
        return [node.demand for node in self.nodes]

    @property
    def arc_list(self):
        """List of arcs as (src, dst) pairs."""
        return [
            (int(src), int(dst))
            for src, dst in zip(self.arc_index[0], self.arc_index[1])
        ]

    # CHANGE: helper to detect whether TW info exists in the instance
    def has_time_windows(self):
        """Return True if at least one node carries time window information."""
        return any(
            (getattr(n, "ready_time", None) is not None) or (getattr(n, "due_time", None) is not None)
            for n in self.nodes
        )

    def to_dict(self):
        """Convert instance to a serializable dictionary.

        Returns
        -------
        dict
            Dictionary containing all instance attributes needed for saving/loading.
        """
        # CHANGE: set instance_type automatically (cvrp vs cvrptw) based on node TW fields
        instance_type = "cvrptw" if self.has_time_windows() else "cvrp"

        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "arc_index": self.arc_index,
            # "arc_lookup": self.arc_lookup,  # unchanged: typically not serialized
            "vehicle_capacity": self.vehicle_capacity,
            "arc_costs": self.arc_costs,
            "nb_vehicles": self.nb_vehicles,
            "instance_type": instance_type,  # CHANGE: dynamic instance type
        }

    def eval_sol_dict(self, solution):
        """Compute total costs for a solution provided as a dict.

        Parameters
        ----------
        solution : dict
            Solution dictionary with keys (i, j) and values indicating activation (0/1 or flow).

        Returns
        -------
        float or int
            Total costs incurred by the solution.
        """
        return sum(
            v * self.arc_costs[self.arc_lookup[(i, j)]]
            for (i, j), v in solution.items()
            if (i, j) in self.arc_lookup
        )

    def eval_sol_list(self, solution):
        """Compute total costs for a solution provided as a vector aligned with arc_costs.

        Parameters
        ----------
        solution : np.array
            Solution vector with one entry per arc (same ordering as arc_costs).

        Returns
        -------
        float or int
            Total costs incurred by the solution.
        """
        return np.sum(solution * self.arc_costs)

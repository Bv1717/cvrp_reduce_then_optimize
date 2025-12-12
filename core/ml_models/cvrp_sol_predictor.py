""" Wrapper for solution-edge prediction models (Imitation learning / classification). """

from functools import partial

import numpy as np
import torch

from core.ml_models.base_learner import BaseLearner
from core.ml_models.baselines import LinearModel
from core.ml_models.baselines import MLP

# from core.ml_models.bipartite_gnn import GraphNNAtt

##Changes
from core.ml_models.gnn import GraphNNAtt
##End changes


# from core.ml_models.losses import loss_edges_multiclass
# from core.utils.kpi import eval_edge_prediction_accuracy

##Changes
from core.ml_models.losses import loss_arcs_multiclass
from core.utils.kpi import eval_arc_prediction_accuracy
from core.ml_models.losses import FY_loss_regularised
from core.cvrp_solvers.ip_grb import cvrp_subset_connections
import gurobipy as gp
from core.cvrp_solvers.ip_grb import sol_vals
##End changes

from core.utils.kpi import get_accuracy


# class BaseSolEdgePredictor(BaseLearner):
#     """Base learner for solution edge prediction models.

#     Parameters
#     ----------
#     model: pytorch model
#         Pytorch model to be trained.
#     class_weight: float or list
#         Class weights to be used.
#     adam_params: dict, optional
#         Dictionary of Adam parameters.
#     lr_schedule: dict, optional
#         Dictionary of learning rate scheduler parameters.
#     input_transformer: function
#         Transformation function to be applied to input before passing into model.

#     """

#     def __init__(
#         self,
#         model,
#         class_weight=None,
#         adam_params=None,
#         lr_schedule=None,
#         input_transformer=None,
#     ):
#         super(BaseSolEdgePredictor, self).__init__(model, adam_params, lr_schedule)

#         self.multi_class = self.model.output_dim > 1

#         # Binary classification: (weighted) BCE loss
#         self.class_weight = class_weight
#         if not self.multi_class:
#             if self.class_weight is not None:
#                 self.class_weight = torch.FloatTensor([self.class_weight])
#             self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.class_weight)
#             self.evaluate = self.evaluate_binary
#         # Multi-class classification: (weighted) Cross Entropy loss
#         else:
#             if self.class_weight is not None:
#                 self.class_weight = torch.FloatTensor(self.class_weight)
#             self.loss_fn = partial(loss_edges_multiclass, edge_cw=self.class_weight)
#             self.evaluate = self.evaluate_multiclass

#         self.input_transformer = input_transformer

#     def forward_pass(self, inputs, true_edge_matrix):
#         # compute edge predictions
#         edge_predictions_raw, edge_predictions = self.predict_edges(inputs)

#         # compute loss
#         if not self.multi_class:
#             true_edge_matrix = true_edge_matrix.float()
#         loss = self.loss_fn(edge_predictions_raw, true_edge_matrix)

#         return loss, edge_predictions

#     def predict_edges(self, inputs, train=True):
#         """Make edge prediction.

#         Parameters
#         ----------
#         inputs: list
#             List of model inputs.
#         train: bool, optional
#             Indicate whether model should be in training or evaluation mode.

#         """
#         if not train:
#             self.model.eval()

#         x = inputs
#         if self.input_transformer is not None:
#             x = self.input_transformer.transform(x)
#         x = [x_i.float() for x_i in x]

#         predictions_raw = self.model(*x)

#         if self.multi_class:
#             predictions = torch.nn.functional.log_softmax(predictions_raw, dim=-1)
#         else:
#             predictions = torch.sigmoid(predictions_raw)

#         return predictions_raw, predictions

#     def evaluate_binary(self, data_loaders, batch_tensor=True):
#         """Evaluate performance of binary classification model.

#         Parameters
#         ----------
#         data_loaders: dict {<name>: <DataLoader>}
#             Data loaders providing evaluation data.
#         batch_tensor: bool, optional
#             Indicate whether batch is provided as tensor (True, default) or as a list.

#         """

#         self.model.eval()

#         p = dict()
#         for loader_name, data_loader in data_loaders.items():
#             n = len(data_loader)
#             running_loss = 0
#             running_acc = 0
#             running_rec = 0
#             running_prec = 0
#             running_f = 0
#             for batch in data_loader:
#                 if batch_tensor:
#                     x, y = batch
#                     # get predictions
#                     loss, outputs = self.forward_pass(inputs=x, true_edge_matrix=y)
#                     loss = loss.item()
#                     predictions = np.round(outputs.detach().numpy())
#                     # evaluate prediction accuracy
#                     accuracy, recall, precision, fscore = eval_edge_prediction_accuracy(
#                         predictions, y.detach().numpy()
#                     )
#                 else:
#                     loss = 0.0
#                     accuracy = 0.0
#                     recall = 0.0
#                     precision = 0.0
#                     fscore = 0.0
#                     num_batch_samples = 0
#                     for sample in batch:
#                         # print(sample)
#                         num_batch_samples += 1
#                         x, y = sample
#                         sample_loss, sample_outputs = self.forward_pass(
#                             inputs=x, true_edge_matrix=y
#                         )
#                         loss += sample_loss.item()
#                         sample_predictions = np.round(sample_outputs.detach().numpy())
#                         # evaluate prediction accuracy
#                         (
#                             sample_accuracy,
#                             sample_recall,
#                             sample_precision,
#                             sample_fscore,
#                         ) = eval_edge_prediction_accuracy(
#                             sample_predictions, y.detach().numpy()
#                         )
#                         accuracy += sample_accuracy
#                         recall += sample_recall
#                         precision += sample_precision
#                         fscore += sample_fscore
#                     # get batch averages
#                     loss /= float(num_batch_samples)
#                     accuracy /= float(num_batch_samples)
#                     recall /= float(num_batch_samples)
#                     precision /= float(num_batch_samples)
#                     fscore /= float(num_batch_samples)
#                 # collect KPIs
#                 running_loss += loss
#                 running_acc += accuracy
#                 running_rec += recall
#                 running_prec += precision
#                 running_f += fscore

#             p[f"{loader_name}_loss"] = running_loss / n
#             p[f"{loader_name}_accuracy"] = running_acc / n
#             p[f"{loader_name}_recall"] = running_rec / n
#             p[f"{loader_name}_precision"] = running_prec / n
#             p[f"{loader_name}_fscore"] = running_f / n

#         return p

#     def evaluate_multiclass(self, data_loaders, batch_tensor=True):
#         """Evaluate performance multi-class classification model.

#         Parameters
#         ----------
#         data_loaders: dict {<name>: <DataLoader>}
#             Data loaders providing evaluation data.
#         batch_tensor: bool, optional
#             Indicate whether batch is provided as tensor (True, default) or as a list.

#         """

#         self.model.eval()

#         p = dict()
#         for loader_name, data_loader in data_loaders.items():
#             n = len(data_loader)
#             running_loss = 0
#             running_acc = 0
#             for batch in data_loader:
#                 if batch_tensor:
#                     x, y = batch
#                     # get predictions
#                     loss, outputs = self.forward_pass(inputs=x, true_edge_matrix=y)
#                     loss = loss.item()
#                     _, predictions = outputs.max(-1)
#                     predictions = predictions.detach().numpy()
#                     # evaluate prediction accuracy and confusion matrix
#                     accuracy = get_accuracy(predictions, y.detach().numpy())
#                 else:
#                     loss = 0.0
#                     accuracy = 0.0
#                     num_batch_samples = 0
#                     for sample in batch:
#                         # print(sample)
#                         num_batch_samples += 1
#                         x, y = sample
#                         sample_loss, sample_outputs = self.forward_pass(
#                             inputs=x, true_edge_matrix=y
#                         )
#                         loss += sample_loss.item()
#                         _, sample_predictions = sample_outputs.max(-1)
#                         sample_predictions = sample_predictions.detach().numpy()
#                         # evaluate prediction accuracy and confusion matrix
#                         sample_accuracy = get_accuracy(
#                             sample_predictions, y.detach().numpy()
#                         )
#                         accuracy += sample_accuracy
#                     # get batch averages
#                     loss /= float(num_batch_samples)
#                     accuracy /= float(num_batch_samples)
#                 # collect KPIs
#                 running_loss += loss
#                 running_acc += accuracy

#             p[f"{loader_name}_loss"] = running_loss / n
#             p[f"{loader_name}_accuracy"] = running_acc / n

#         return p


# class GCNNSolEdgePredictor(BaseSolEdgePredictor):
#     """GNN-based solution edge predictor."""

#     def __init__(
#         self,
#         model_config,
#         **kwargs,
#     ):

#         input_dims = (
#             model_config.node_dim,
#             model_config.node_dim,
#             model_config.edge_dim,
#         )
#         hidden_dim = model_config.hidden_layer_dim
#         num_conv_layers = model_config.num_conv_layers
#         num_dense_layers = model_config.num_dense_layers

#         conv_dims = [
#             (hidden_dim, hidden_dim, hidden_dim) for _ in range(num_conv_layers)
#         ]
#         dense_dims = [hidden_dim for _ in range(num_dense_layers)]
#         output_dim = model_config.edge_output_dim

#         model = GraphNNAtt(
#             input_dims,
#             conv_dims,
#             dense_dims,
#             output_dim,
#         )

#         super(GCNNSolEdgePredictor, self).__init__(model, **kwargs)


# class EdgeLogRegSolEdgePredictor(BaseSolEdgePredictor):
#     """LogReg-based solution edge predictor."""

#     def __init__(self, model_config, **kwargs):

#         input_dim = model_config.edge_dim
#         output_dim = model_config.edge_output_dim
#         model = LinearModel(input_dim, output_dim)

#         super(EdgeLogRegSolEdgePredictor, self).__init__(model, **kwargs)


# class EdgeMLPSolEdgePredictor(BaseSolEdgePredictor):
#     """MLP-based solution edge predictor."""

#     def __init__(self, model_config, **kwargs):

#         input_dim = model_config.edge_dim
#         hidden_dims = [
#             model_config.hidden_layer_dim for _ in range(model_config.num_dense_layers)
#         ]
#         output_dim = model_config.edge_output_dim
#         model = MLP(input_dim, hidden_dims, output_dim)

#         super(EdgeMLPSolEdgePredictor, self).__init__(model, **kwargs)



##Changes

class BaseSolArcPredictor(BaseLearner):
    """Base learner for solution arc prediction models.

    Parameters
    ----------
    model: pytorch model
        Pytorch model to be trained.
    class_weight: float or list
        Class weights to be used.
    adam_params: dict, optional
        Dictionary of Adam parameters.
    lr_schedule: dict, optional
        Dictionary of learning rate scheduler parameters.
    input_transformer: function
        Transformation function to be applied to input before passing into model.

    """

    def __init__(
        self,
        model,
        class_weight=None,
        adam_params=None,
        lr_schedule=None,
        input_transformer=None,
    ):
        super(BaseSolArcPredictor, self).__init__(model, adam_params, lr_schedule)

        self.multi_class = self.model.output_dim > 1

        # Binary classification: (weighted) BCE loss
        self.class_weight = class_weight
        if not self.multi_class:
            if self.class_weight is not None:
                self.class_weight = torch.FloatTensor([self.class_weight])
            self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.class_weight)
            self.loss_FY = FY_loss_regularised
            self.evaluate = self.evaluate_binary
        # Multi-class classification: (weighted) Cross Entropy loss
        else:
            if self.class_weight is not None:
                self.class_weight = torch.FloatTensor(self.class_weight)
            self.loss_fn = partial(loss_arcs_multiclass, arc_cw=self.class_weight)
            self.evaluate = self.evaluate_multiclass

        self.input_transformer = input_transformer

    def forward_pass(self, data, FW_env, k, regul_lambda , max_iterations):

        arc_predictions_raw, arc_predictions = self.predict_arcs(data)

        # print("arc_predictions : ", arc_predictions)

        instances = data.to_data_list()

        edge_pos_instance = 0
        true_arc_list = data.y.float() if not self.multi_class else data.y

        total_loss = 0

        true_total_loss = 0

        y_hat_predictions = []


        for i, instance in enumerate(instances):
            arcs_list = [(int(src) , int(dst)) for src, dst in zip(instance.edge_index[0], instance.edge_index[1])]

            true_arc_solution = true_arc_list[edge_pos_instance : instance.num_edges + edge_pos_instance].squeeze().tolist()
            true_arc_solution_tensor = true_arc_list[edge_pos_instance : instance.num_edges + edge_pos_instance].squeeze()
            predicted_costs_list = arc_predictions_raw[edge_pos_instance : instance.num_edges + edge_pos_instance].squeeze().tolist()
            predicted_costs_tensor = arc_predictions_raw[
                edge_pos_instance : edge_pos_instance + instance.num_edges
            ].squeeze()  
            edge_pos_instance += instance.num_edges 

            demands_list = instance.demands.squeeze().tolist()

            # relevant_connections = [True]*len(arcs_list)

            # solution = {}

            # model, x, _ = cvrp_subset_connections(
            #     instance.demands.squeeze().tolist(),
            #     instance.edge_index,
            #     predicted_costs_list,
            #     int(instance.nb_vehicles.item()),
            #     int(instance.vehicle_capacity.item()),
            #     relevant_connections,
            #     relax=False,
            # )
            # model.setParam("OutputFlag", 0)
            # model.setParam("TimeLimit", 0.2)
            # model.optimize()
            # if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.SUBOPTIMAL  or model.SolCount>0:
            #     # print("Number of solutions found:", model.SolCount)
            #     solution = sol_vals(x)
            #     # print("Optimal objective value with restriction:", model.objVal)
            # else:
            #     # print("Number of solutions found:", model.SolCount)
            #     # print("No feasible solution found, status:", model.status)
            #     solution = {}


            # test_value = regul_lambda/2 * torch.sum(true_arc_solution_tensor**2) - torch.dot(predicted_costs_tensor, true_arc_solution_tensor)
            # print("true_value :", test_value.item())
            # # solution, _ = heu_solve_HGS_VRP(instance.demands.squeeze().tolist(), instance.edge_index, 
            # #           predicted_costs_list, int(instance.nb_vehicles.item()),
            # #             int(instance.vehicle_capacity.item()), relevant_connections)
            
            # sol_list = []
            # for src, tgt in zip(instance.edge_index[0], instance.edge_index[1]):
            #     val = solution.get((src, tgt), 0.0)  # default 0 if missing
            #     sol_list.append(val)

            # gradient_cost_list = [0]*len(predicted_costs_list)
            # for t in range(max_iterations):

            #     print("test_value : ",
            #     regul_lambda/2 * sum(y_val**2 for y_val in sol_list)
            #     - np.dot(predicted_costs_list, sol_list))

            #     for i in range(len(gradient_cost_list)):
            #         gradient_cost_list[i] = regul_lambda*sol_list[i] - predicted_costs_list[i]

            #     model, x, _ = cvrp_subset_connections(
            #         instance.demands.squeeze().tolist(),
            #         instance.edge_index,
            #         gradient_cost_list,
            #     int(instance.nb_vehicles.item()),
            #     int(instance.vehicle_capacity.item()),
            #         relevant_connections,
            #         relax=False,
            #     )
            #     model.setParam("OutputFlag", 0)
            #     model.setParam("TimeLimit",1)
            #     model.optimize()
            #     print("Number of solutions found:", model.SolCount)
            #     if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.SUBOPTIMAL or model.SolCount>0:
            #         solution = sol_vals(x)
            #         # print("Optimal objective value with restriction:", model.objVal)
            #     else:
            #         # print("No feasible solution found, status:", model.status)
            #         solution = {}
 
            #     tempora_sol_list = []

            #     for src, tgt in arcs_list:
            #         val = solution.get((src, tgt), 2)  # default 0 if missing
            #         tempora_sol_list.append(val)  

            #     for i in range(len(gradient_cost_list)):
            #         sol_list[i] = sol_list[i] + 2/(t+3)*(tempora_sol_list[i] - sol_list[i])




            # print("final_value : ",
            #         regul_lambda/2 * sum(y_val**2 for y_val in sol_list)
            #         - np.dot(predicted_costs_list, sol_list))


            y_hat = self.loss_FY(FW_env, predicted_costs_list,
                                    demands_list, instance.edge_index, int(instance.nb_vehicles.item()),
                                    int(instance.vehicle_capacity.item()), true_arc_solution,
                                    regul_lambda=regul_lambda, max_iterations= max_iterations)
            
        
            y_hat_dict = dict(y_hat)

            # print("y_hat_dict : ",y_hat_dict)

            y_hat_list = [y_hat_dict.get(tuple(arc), 0.0) for arc in arcs_list]

            # y_hat_list = sol_list


            y_hat_tensor = torch.tensor(y_hat_list, dtype=torch.float32, device=predicted_costs_tensor.device)
            # print("y_hat_tensor :", y_hat_tensor)
            print("y_hat_tensor", y_hat.shape())

            y_pre_prediction = (y_hat_tensor / (y_hat_tensor.sum() + 1e-8)).unsqueeze(1)
            # y_hat_predictions.append((y_hat_tensor / (y_hat_tensor.sum() + 1e-8)).unsqueeze(1))

            topk_indices = torch.topk(y_pre_prediction.squeeze(), k).indices
            predictions = torch.zeros_like(y_pre_prediction, dtype=torch.int)
            predictions[topk_indices] = 1
            y_hat_predictions.append(predictions)

            # Squared norm
            y_hat_squared_norm = torch.norm(y_hat_tensor).pow(2).item()

            y_true_squared_norm = torch.sum(true_arc_solution_tensor.squeeze()**2).detach().cpu().numpy()

            correct_objective = torch.sum(predicted_costs_tensor * true_arc_solution_tensor)
            predicted_objective = torch.sum(predicted_costs_tensor * y_hat_tensor)



            # Special loss: difference between predicted and correct objectives
            loss_instance = predicted_objective - correct_objective

            # print("obj_value_FW : ", obj_value)
            # print("FW_obj_value : ", -predicted_objective.detach().cpu().numpy() + 2.5*y_hat_squared_norm)

            true_loss_instance = loss_instance.detach().cpu().numpy() - (regul_lambda/2)*(y_hat_squared_norm-y_true_squared_norm)
            total_loss += loss_instance

            true_total_loss += true_loss_instance
            # print("true_loss_instance : ", true_loss_instance)

        total_loss = total_loss/len(instances)
        true_total_loss = true_total_loss/len(instances)
        
        # loss = self.loss_fn(arc_predictions_raw, true_arc_list)

        y_hat_predictions_tensor = torch.cat(y_hat_predictions, dim=0)

        # print(y_hat_predictions_tensor)

        return total_loss, arc_predictions, true_total_loss

    def predict_arcs(self, data, train=True):
        """Make arc prediction.

        Parameters
        ----------
         data : torch_geometric.data.Data or torch_geometric.data.Batch
        A single graph (Data) or a batch of graphs (Batch) containing:
        - data.x : node feature matrix [num_nodes, node_dim]
        - data.edge_index : edge connectivity [2, num_edges]
        - data.edge_attr : edge feature matrix [num_edges, edge_dim]
        - data.y : target labels (optional, used for training/evaluation)

        train : bool, optional
            If False, the model is set to evaluation mode before prediction.

        Returns
        -------
        predictions_raw : torch.Tensor
            Raw model outputs (logits).
        predictions : torch.Tensor
            Post-processed predictions (sigmoid for binary, log_softmax for multi-class).
        """
        if not train:
            self.model.eval()

        if self.input_transformer is not None:

            x_norm, edge_attr_norm = self.input_transformer.transform([data.x, data.edge_attr])

            data.x, data.edge_attr = x_norm, edge_attr_norm

        predictions_raw = self.model(data.x, data.edge_attr, data.edge_index)

        if self.multi_class:
            predictions = torch.nn.functional.log_softmax(predictions_raw, dim=-1)
        else:
            predictions = torch.sigmoid(predictions_raw)

        return predictions_raw, predictions

    def evaluate_binary(self, data_loaders, FW_env, top_k, regul_lambda, max_iterations):
        """Evaluate performance of binary classification model.

        Parameters
        ----------
        data_loaders: dict {<name>: <DataLoader>}
            Data loaders providing evaluation data.

        """

        self.model.eval()

        p = dict()
        for loader_name, data_loader in data_loaders.items():
            n = len(data_loader)
            running_loss = 0
            running_acc = 0
            running_rec = 0
            running_prec = 0
            running_f = 0
            for batch in data_loader:
                loss, outputs, true_total_loss = self.forward_pass(batch, FW_env, top_k, 
                                                regul_lambda, max_iterations)
                loss = loss.item()

                threshold = 0.5
                # predictions = (outputs > threshold).int().detach().cpu().numpy()

                # k = 50*32 # or based on vehicle capacity
                # topk_indices = torch.topk(outputs.squeeze(), k).indices
                # predictions = torch.zeros_like(outputs, dtype=torch.int).detach().cpu().numpy()
                # predictions[topk_indices] = 1

                outputs_norm = (outputs - outputs.min()) / (outputs.max() - outputs.min() + 1e-8)
                predictions = (outputs_norm > 0.4).int().detach().cpu().numpy()

                # predictions = outputs.detach().cpu().numpy()

                accuracy, recall, precision, fscore = eval_arc_prediction_accuracy(
                    predictions, batch.y.cpu().numpy()
                )
                # collect KPIs
                # running_loss += loss
                running_loss +=true_total_loss
                running_acc += accuracy
                running_rec += recall
                running_prec += precision
                running_f += fscore

            p[f"{loader_name}_loss"] = running_loss / n
            p[f"{loader_name}_accuracy"] = running_acc / n
            p[f"{loader_name}_recall"] = running_rec / n
            p[f"{loader_name}_precision"] = running_prec / n
            p[f"{loader_name}_fscore"] = running_f / n

        return p

    def evaluate_multiclass(self, data_loaders):
        """Evaluate performance multi-class classification model.

        Parameters
        ----------
        data_loaders: dict {<name>: <DataLoader>}
            Data loaders providing evaluation data.

        """

        self.model.eval()

        p = dict()
        for loader_name, data_loader in data_loaders.items():
            n = len(data_loader)
            running_loss = 0
            running_acc = 0
            for batch in data_loader:
                loss, outputs = self.forward_pass(batch)
                loss = loss.item()
                _, predictions = outputs.max(-1)
                predictions = predictions.cpu().numpy()
                accuracy = get_accuracy(predictions, batch.y.cpu().numpy())
                running_loss += loss
                running_acc += accuracy
            p[f"{loader_name}_loss"] = running_loss / n
            p[f"{loader_name}_accuracy"] = running_acc / n

        return p
    
class GCNNSolArcPredictor(BaseSolArcPredictor):
    """GNN-based solution arc predictor."""

    def __init__(
        self,
        model_config,
        **kwargs,
    ):

        input_dims = (
            model_config.node_dim,
            model_config.arc_dim,
        )
        hidden_dim = model_config.hidden_layer_dim
        num_conv_layers = model_config.num_conv_layers
        num_dense_layers = model_config.num_dense_layers

        conv_dims = [
            (hidden_dim, hidden_dim) for _ in range(num_conv_layers)
        ]
        dense_dims = [hidden_dim for _ in range(num_dense_layers)]
        output_dim = model_config.arc_output_dim

        model = GraphNNAtt(
            input_dims,
            conv_dims,
            dense_dims,
            output_dim,
        )

        super(GCNNSolArcPredictor, self).__init__(model, **kwargs)

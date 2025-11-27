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
            self.evaluate = self.evaluate_binary
        # Multi-class classification: (weighted) Cross Entropy loss
        else:
            if self.class_weight is not None:
                self.class_weight = torch.FloatTensor(self.class_weight)
            self.loss_fn = partial(loss_arcs_multiclass, arc_cw=self.class_weight)
            self.evaluate = self.evaluate_multiclass

        self.input_transformer = input_transformer

    def forward_pass(self, data):

        # instances = data.to_data_list()
        # print(f"Number of instances in batch: {len(instances)}")

        # for i, instance in enumerate(instances):
        #     print(f"\n--- Instance {i} ---")
        #     print("Nodes:", instance.num_nodes)
        #     print("Edges:", instance.num_edges)
        #     print("Node features shape:", instance.x.shape)
        #     print("Edge index:", instance.edge_index)
        #     print("Labels:", instance.y)
        #     if hasattr(instance, "num_vehicles"):
        #         print("Num vehicles:", instance.num_vehicles.item())
        #     if hasattr(instance, "vehicle_capacity"):
        #         print("Vehicle capacity:", instance.vehicle_capacity.item())
        arc_predictions_raw, arc_predictions = self.predict_arcs(data)
        true_arc_list = data.y.float() if not self.multi_class else data.y
        
        loss = self.loss_fn(arc_predictions_raw, true_arc_list)

        return loss, arc_predictions

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

    def evaluate_binary(self, data_loaders):
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
                loss, outputs = self.forward_pass(batch)
                loss = loss.item()

                threshold = 0.5
                predictions = (outputs > threshold).int().detach().cpu().numpy()

                accuracy, recall, precision, fscore = eval_arc_prediction_accuracy(
                    predictions, batch.y.cpu().numpy()
                )
                # collect KPIs
                running_loss += loss
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

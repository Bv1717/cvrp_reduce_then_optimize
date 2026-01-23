""" Training script for solution edge prediction model (supervised with binary labels)."""

from datetime import datetime
from functools import partial
import logging
import os

import hydra
import numpy as np
from omegaconf import DictConfig
from omegaconf import open_dict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import torch

# from core.ml_models.fctp_sol_predictor import GCNNSolEdgePredictor
# from core.ml_models.fctp_sol_predictor import (
#     EdgeLogRegSolEdgePredictor,
# )
# from core.ml_models.fctp_sol_predictor import (
#     EdgeMLPSolEdgePredictor,
# )
# from core.utils.ml_utils import FCTPData

##Changes
from core.ml_models.cvrp_sol_predictor import GCNNSolArcPredictor
from core.utils.ml_utils import CVRPData, LazyCVRPData
from core.utils.ml_utils import get_graph_raw_features_for_instance
from core.utils.ml_utils import fit_cvrp_standard_normalizer_from_paths, get_cvrp_class_weight_from_paths
from core.data_processing.data_utils import load_sample
##End changes

from core.utils.ml_utils import StandardNormalizer
from core.utils.ml_utils import MultiInputNormalizer



# from core.utils.ml_utils import get_bipartite_raw_features
# from core.utils.ml_utils import get_edge_features
# from core.utils.ml_utils import get_bipartite_advanced_edge_features

##Changes
from core.utils.ml_utils import get_graph_raw_features
from core.utils.ml_utils import get_arc_features
##End changes

from core.utils.ml_utils import get_raw_targets
from core.utils.ml_utils import get_sample_paths
from core.utils.ml_utils import get_class_weights
from core.utils.ml_utils import training_wrapper
from core.utils.ml_utils import binary_classification_eval_display_func
from core.utils.ml_utils import multiclass_classification_eval_display_func
from core.evaluation.ml_model_evaluation import select_best_model_across_candidates


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "configs", "training"),
    config_name="config",
)
# def main(training_config: DictConfig) -> None:

#     # Set random seeds
#     seed = training_config.seed
#     np.random.seed(seed)
#     torch.manual_seed(seed)

#     # Extract model config
#     model_config = training_config.model
#     model_spec = "_".join([f"{k}_{v}" for k, v in model_config.items()])

#     # Prepare output directory
#     out_dir = os.path.join(training_config.out_dir, model_spec)
#     if training_config.cross_validate:
#         out_dir = os.path.join(out_dir, "cross_val")
#     else:
#         out_dir = os.path.join(out_dir, "application")
#     os.makedirs(out_dir, exist_ok=True)

#     # Set up logger
#     os.makedirs(training_config.log_dir, exist_ok=True)
#     logger = logging.getLogger()
#     logger.addHandler(
#         logging.FileHandler(
#             os.path.join(
#                 training_config.log_dir,
#                 f"{datetime.now().strftime('%Y%m%d_%H:%M:%S')}_train.log",
#             ),
#             mode="w",
#         )
#     )

#     logger.info(f"Training parameters: {training_config}")

#     # Get list of (subset of) sample file paths
#     sample_paths = get_sample_paths(
#         training_config.data_path, training_config.num_samples
#     )

#     #######################################################
#     # Extract features
#     #######################################################

#     if model_config.features == "bipartite_raw":
#         x_supply, x_demand, x_connections = get_bipartite_raw_features(sample_paths)
#         with open_dict(model_config):
#             model_config.node_dim = x_supply[0].shape[-1]
#             model_config.edge_dim = x_connections[0].shape[-1]
#         x = (x_supply, x_demand, x_connections)
#     elif model_config.features in [
#         "combined_raw_edge_features",
#         "advanced_edge_features",
#         "advanced_edge_features_plus_stat_features",
#     ]:
#         x_connections = get_edge_features(
#             sample_paths,
#             features=model_config.features,
#         )
#         with open_dict(model_config):
#             model_config.edge_dim = x_connections[0].shape[-1]
#         x = (x_connections,)
#     elif model_config.features in [
#         "bipartite_advanced_edge_features",
#         "bipartite_advanced_edge_features_plus_stat_features",
#     ]:
#         x_supply, x_demand, x_connections = get_bipartite_advanced_edge_features(
#             sample_paths,
#             features=model_config.features,
#         )
#         with open_dict(model_config):
#             model_config.node_dim = x_supply[0].shape[-1]
#             model_config.edge_dim = x_connections[0].shape[-1]
#         x = (x_supply, x_demand, x_connections)

#     #######################################################
#     # Extract labels
#     #######################################################

#     if model_config.prediction_task == "binary_classification":
#         y = get_raw_targets(
#             sample_paths,
#             binary_target=True,
#             output_dim=True,
#         )
#         with open_dict(model_config):
#             model_config.edge_output_dim = 1
#     else:
#         raise ValueError

#     #######################################################
#     # Define normalization
#     #######################################################

#     input_transformer = None
#     if model_config.normalization == "standard":
#         normalizers = tuple([StandardNormalizer() for _ in range(len(x))])
#         input_transformer = MultiInputNormalizer(normalizers)

#     #######################################################
#     # Define policy
#     #######################################################

#     if model_config.model == "gcnn":
#         policy_fun = GCNNSolEdgePredictor
#     elif model_config.model == "linear_logreg":
#         policy_fun = EdgeLogRegSolEdgePredictor
#     elif model_config.model == "edge_mlp":
#         policy_fun = EdgeMLPSolEdgePredictor
#     else:
#         raise ValueError

#     #######################################################
#     # Define class weighting
#     #######################################################

#     class_weight_fun = lambda x: None
#     if model_config.prediction_task == "binary_classification":
#         class_weight_fun = partial(get_class_weights, binary=True)

#     #######################################################
#     # Define optimizer
#     #######################################################

#     adam_params = {
#         "lr": training_config.learning_rate,
#         "betas": (training_config.momentum, 0.999),
#         "weight_decay": training_config.weight_decay,
#     }
#     if training_config.lr_decay:
#         lr_schedule = {
#             "factor": training_config.lr_decay_factor,
#             "patience": training_config.patience,
#             "threshold": training_config.opt_threshold,
#         }
#     else:
#         lr_schedule = None

#     #######################################################
#     # Configure training performance logging
#     #######################################################

#     initial_eval_display_func = None
#     running_eval_display_func = None
#     if model_config.prediction_task == "binary_classification":
#         initial_eval_display_func = partial(
#             binary_classification_eval_display_func, include_train=True
#         )
#         running_eval_display_func = partial(
#             binary_classification_eval_display_func, include_train=False
#         )
#     elif model_config.prediction_task in ["threeclass_classification"]:
#         initial_eval_display_func = partial(
#             multiclass_classification_eval_display_func, include_train=True
#         )
#         running_eval_display_func = partial(
#             multiclass_classification_eval_display_func, include_train=False
#         )

#     #######################################################
#     # Training
#     #######################################################

#     if not training_config.cross_validate:

#         # randomly split data set into training and validation data
#         split = train_test_split(
#             *x, y, test_size=training_config.test_split, random_state=0
#         )
#         train_data = [split[i] for i in range(len(split)) if i % 2 == 0]
#         val_data = [split[i] for i in range(len(split)) if i % 2 == 1]
#         x_train, y_train = train_data[:-1], train_data[-1]
#         x_val, y_val = val_data[:-1], val_data[-1]

#         train_dataset = FCTPData(x_train, y_train)
#         val_dataset = FCTPData(x_val, y_val)

#         if input_transformer is not None:
#             input_transformer.fit(x_train)

#         policy = policy_fun(
#             model_config=model_config,
#             adam_params=adam_params,
#             lr_schedule=lr_schedule,
#             input_transformer=input_transformer,
#             class_weight=class_weight_fun(y_train),
#         )

#         training_wrapper(
#             training_config,
#             model_config,
#             train_dataset,
#             val_dataset,
#             policy,
#             out_dir,
#             logger,
#             eval_train_data=False,
#             initial_eval_display_func=initial_eval_display_func,
#             running_eval_display_func=running_eval_display_func,
#         )

#     else:
#         kf = KFold(n_splits=5, shuffle=True, random_state=0)
#         kf.get_n_splits(y)
#         for i, (train_index, val_index) in enumerate(kf.split(y)):
#             logger.info(f"Training on fold {i}")
#             fold_out_dir = os.path.join(out_dir, f"fold_{i}")
#             os.makedirs(fold_out_dir, exist_ok=True)

#             x_train, y_train = tuple([x_i[train_index] for x_i in x]), y[train_index]
#             x_val, y_val = tuple([x_i[val_index] for x_i in x]), y[val_index]

#             train_dataset = FCTPData(x_train, y_train)
#             val_dataset = FCTPData(x_val, y_val)

#             if input_transformer is not None:
#                 input_transformer.fit(x_train)

#             policy = policy_fun(
#                 model_config=model_config,
#                 adam_params=adam_params,
#                 lr_schedule=lr_schedule,
#                 input_transformer=input_transformer,
#                 class_weight=class_weight_fun(y_train),
#             )

#             training_wrapper(
#                 training_config,
#                 model_config,
#                 train_dataset,
#                 val_dataset,
#                 policy,
#                 fold_out_dir,
#                 logger,
#                 eval_train_data=False,
#                 initial_eval_display_func=initial_eval_display_func,
#                 running_eval_display_func=running_eval_display_func,
#             )
#         logger.info("Select best model across folds.")
#         select_best_model_across_candidates(out_dir)


# if __name__ == "__main__":
#     main()


##Changes

def main(training_config: DictConfig) -> None:

    # Set random seeds
    seed = training_config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Extract model config
    model_config = training_config.model
    model_spec = "_".join([f"{k}_{v}" for k, v in model_config.items()])

    # Prepare output directory
    out_dir = os.path.join(training_config.out_dir, model_spec)
    if training_config.cross_validate:
        out_dir = os.path.join(out_dir, "cross_val")
    else:
        out_dir = os.path.join(out_dir, "application")
    os.makedirs(out_dir, exist_ok=True)

    # Set up logger
    os.makedirs(training_config.log_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(
            os.path.join(
                training_config.log_dir,
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_train.log",
            ),
            mode="w",
        )
    )

    logger.info(f"Training parameters: {training_config}")

    # Get list of (subset of) sample file paths
    sample_paths = get_sample_paths(
        training_config.data_path, training_config.num_samples
    )

    use_lazy = getattr(training_config, "lazy_loading", False)

    #######################################################
    # Extract features
    #######################################################

    if model_config.features == "graph_raw":
        if use_lazy:
            # Only infer feature dimensions from a single sample (no pre-loading of all samples)
            s0 = load_sample(sample_paths[0])
            x_no0, x_a0, _, _, _ = get_graph_raw_features_for_instance(s0["instance"], batch_dim=False)
            with open_dict(model_config):
                model_config.node_dim = x_no0.shape[-1]
                model_config.arc_dim = x_a0.shape[-1]
        else:
            x_nodes, x_connections, arc_index, nb_vehicles, vehicle_capacity = get_graph_raw_features(sample_paths)
            with open_dict(model_config):
                model_config.node_dim = x_nodes[0].shape[-1]
                model_config.arc_dim = x_connections[0].shape[-1]
            x_weights = np.zeros(1)
            x = (x_nodes, x_connections, arc_index, nb_vehicles, vehicle_capacity, x_weights)
    else:
        raise ValueError

    #######################################################
    # Extract labels
    #######################################################

    if model_config.prediction_task == "binary_classification":
        with open_dict(model_config):
            model_config.arc_output_dim = 1

        # Lazy mode: labels dataset içinde __getitem__'te üretilecek, burada y precompute etmiyoruz.
        if not use_lazy:
            y = get_raw_targets(
                sample_paths,
                binary_target=True,
                output_dim=True,
            )
    else:
        raise ValueError(f"Unsupported prediction_task: {model_config.prediction_task}")


    #######################################################
    # Define normalization
    #######################################################

    input_transformer = None  # fitted after train/val split (needs train split statistics)

    #######################################################
    # Define policy
    #######################################################

    if model_config.model == "gcnn":
        policy_fun = GCNNSolArcPredictor
    else:
        raise ValueError

    #######################################################
    # Define class weighting
    #######################################################

    class_weight_fun = lambda x: None
    if model_config.prediction_task == "binary_classification":
        class_weight_fun = partial(get_class_weights, binary=True)

    #######################################################
    # Define optimizer
    #######################################################

    adam_params = {
        "lr": training_config.learning_rate,
        "betas": (training_config.momentum, 0.999),
        "weight_decay": training_config.weight_decay,
    }
    if training_config.lr_decay:
        lr_schedule = {
            "factor": training_config.lr_decay_factor,
            "patience": training_config.patience,
            "threshold": training_config.opt_threshold,
        }
    else:
        lr_schedule = None

    #######################################################
    # Configure training performance logging
    #######################################################

    initial_eval_display_func = None
    running_eval_display_func = None
    if model_config.prediction_task == "binary_classification":
        initial_eval_display_func = partial(
            binary_classification_eval_display_func, include_train=True
        )
        running_eval_display_func = partial(
            binary_classification_eval_display_func, include_train=False
        )
    elif model_config.prediction_task in ["threeclass_classification"]:
        initial_eval_display_func = partial(
            multiclass_classification_eval_display_func, include_train=True
        )
        running_eval_display_func = partial(
            multiclass_classification_eval_display_func, include_train=False
        )

    #######################################################
    # Training
    #######################################################

    if not training_config.cross_validate:

        # randomly split data set into training and validation data
        # split = train_test_split(
        #     *x, y, test_size=training_config.test_split, random_state=0
        # )
        # train_data = [split[i] for i in range(len(split)) if i % 2 == 0]
        # val_data = [split[i] for i in range(len(split)) if i % 2 == 1]
        # x_train, y_train = train_data[:-1], train_data[-1]
        # x_val, y_val = val_data[:-1], val_data[-1]

        # Split by sample indices (works for both lazy and non-lazy)
        indices = list(range(len(sample_paths)))
        train_idx, val_idx = train_test_split(
            indices, test_size=training_config.test_split, random_state=0
        )

        train_paths = [sample_paths[i] for i in train_idx]
        val_paths   = [sample_paths[i] for i in val_idx]

        if use_lazy:
            train_dataset = LazyCVRPData(train_paths)
            val_dataset   = LazyCVRPData(val_paths)
        else:
            x_nodes_train = [x_nodes[i] for i in train_idx]
            x_connections_train = [x_connections[i] for i in train_idx]
            arc_index_train = [arc_index[i] for i in train_idx]
            nb_vehicles_train = [nb_vehicles[i] for i in train_idx]
            vehicle_capacity_train = [vehicle_capacity[i] for i in train_idx]
            x_weights_train = [x_weights[0] for _ in train_idx]
            y_train = [y[i] for i in train_idx]

            x_nodes_val = [x_nodes[i] for i in val_idx]
            x_connections_val = [x_connections[i] for i in val_idx]
            arc_index_val = [arc_index[i] for i in val_idx]
            nb_vehicles_val = [nb_vehicles[i] for i in val_idx]
            vehicle_capacity_val = [vehicle_capacity[i] for i in val_idx]
            x_weights_val = [x_weights[0] for _ in val_idx]
            y_val = [y[i] for i in val_idx]

            train_dataset = CVRPData(
                (x_nodes_train, x_connections_train, arc_index_train,
                 nb_vehicles_train, vehicle_capacity_train, x_weights_train),
                y_train
            )
            val_dataset = CVRPData(
                (x_nodes_val, x_connections_val, arc_index_val,
                 nb_vehicles_val, vehicle_capacity_val, x_weights_val),
                y_val
            )

        # Make normalization + class weights identical across lazy and non-lazy:
        # compute them from the TRAIN SPLIT PATHS (one scan, no feature pre-loading).
        if model_config.normalization == "standard":
            input_transformer = fit_cvrp_standard_normalizer_from_paths(
                train_paths, feature_fun=get_graph_raw_features_for_instance
            )
        else:
            input_transformer = None

        cw = None
        if model_config.prediction_task == "binary_classification":
            cw = get_cvrp_class_weight_from_paths(train_paths)

        policy = policy_fun(
            model_config=model_config,
            adam_params=adam_params,
            lr_schedule=lr_schedule,
            input_transformer=input_transformer,
            class_weight=class_weight_fun(y_train),
        )

        training_wrapper(
            training_config,
            model_config,
            train_dataset,
            val_dataset,
            policy,
            out_dir,
            logger,
            eval_train_data=False,
            initial_eval_display_func=initial_eval_display_func,
            running_eval_display_func=running_eval_display_func,
        )

    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        kf.get_n_splits(sample_paths)

        for fold_i, (train_idx, val_idx) in enumerate(kf.split(sample_paths)):
            logger.info(f"Training on fold {fold_i}")
            fold_out_dir = os.path.join(out_dir, f"fold_{fold_i}")
            os.makedirs(fold_out_dir, exist_ok=True)

            train_paths = [sample_paths[i] for i in train_idx]
            val_paths   = [sample_paths[i] for i in val_idx]

            if use_lazy:
                train_dataset = LazyCVRPData(train_paths)
                val_dataset   = LazyCVRPData(val_paths)
            else:
                x_nodes_train = [x_nodes[i] for i in train_idx]
                x_connections_train = [x_connections[i] for i in train_idx]
                arc_index_train = [arc_index[i] for i in train_idx]
                nb_vehicles_train = [nb_vehicles[i] for i in train_idx]
                vehicle_capacity_train = [vehicle_capacity[i] for i in train_idx]
                x_weights_train = [x_weights[0] for _ in train_idx]
                y_train = [y[i] for i in train_idx]

                x_nodes_val = [x_nodes[i] for i in val_idx]
                x_connections_val = [x_connections[i] for i in val_idx]
                arc_index_val = [arc_index[i] for i in val_idx]
                nb_vehicles_val = [nb_vehicles[i] for i in val_idx]
                vehicle_capacity_val = [vehicle_capacity[i] for i in val_idx]
                x_weights_val = [x_weights[0] for _ in val_idx]
                y_val = [y[i] for i in val_idx]

                train_dataset = CVRPData(
                    (x_nodes_train, x_connections_train, arc_index_train,
                     nb_vehicles_train, vehicle_capacity_train, x_weights_train),
                    y_train
                )
                val_dataset = CVRPData(
                    (x_nodes_val, x_connections_val, arc_index_val,
                     nb_vehicles_val, vehicle_capacity_val, x_weights_val),
                    y_val
                )

            if model_config.normalization == "standard":
                input_transformer = fit_cvrp_standard_normalizer_from_paths(
                    train_paths, feature_fun=get_graph_raw_features_for_instance
                )
            else:
                input_transformer = None

            cw = None
            if model_config.prediction_task == "binary_classification":
                cw = get_cvrp_class_weight_from_paths(train_paths)

            policy = policy_fun(
                model_config=model_config,
                adam_params=adam_params,
                lr_schedule=lr_schedule,
                input_transformer=input_transformer,
                class_weight=cw,
            )

            training_wrapper(
                training_config,
                model_config,
                train_dataset,
                val_dataset,
                policy,
                fold_out_dir,
                logger,
                eval_train_data=False,
                initial_eval_display_func=initial_eval_display_func,
                running_eval_display_func=running_eval_display_func,
            )
        logger.info("Select best model across folds.")
        select_best_model_across_candidates(out_dir)


if __name__ == "__main__":
    main()
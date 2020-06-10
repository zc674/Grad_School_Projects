from math import sqrt

import torch

from ts.n_beats.model import BLOCK_TYPE
from ts.utils.helper_funcs import SAVE_LOAD_TYPE


def get_config(interval):
    config = {
        "prod": True,
        "device": ("cuda" if torch.cuda.is_available() else "cpu"),
        "percentile": 50,
        "training_percentile": 45,
        "num_of_train_epochs": 100,
        "num_of_train_epochs_sampling": 100,
        "num_of_categories": 6,  # in data provided
        "batch_size": 1024,
        "gradient_clipping": 20,
        "print_output_stats": 3,
        "sample": True,
        "reload": SAVE_LOAD_TYPE.NO_ACTION,
        "add_run_id": False,
        "save_model": SAVE_LOAD_TYPE.NO_ACTION,
        "plot_ts": True,
    }

    if interval == "Quarterly":
        config.update({
            "stack_types": [BLOCK_TYPE.TREND, BLOCK_TYPE.SEASONALITY],
            "thetas_dims": [2, 8],
            "nb_blocks_per_stack": 3,
            "hidden_layer_units": 256,
            "share_weights_in_stack": True,
            "variable": "Quarterly",
            "seasonality": 4,
            "output_size": 8,
            "learning_rate": 1e-1,
            "learning_rates": ((10, 1e-4)),
            "lr_anneal_rate": 0.5,
            "lr_anneal_step": 5,
            "min_learning_rate": 0.0001,
            "lr_ratio": sqrt(10),
            "lr_tolerance_multip": 1.005,
            "min_epochs_before_changing_lrate": 2,
            "dropout": 0.2,
            #"sample_ids": [],
            "sample_ids": ["Q66"],
        })
    elif interval == "Monthly":
        config.update({
            "stack_types": [BLOCK_TYPE.TREND, BLOCK_TYPE.SEASONALITY],
            "thetas_dims": [2, 8],
            "nb_blocks_per_stack": 3,
            "hidden_layer_units": 256,
            "share_weights_in_stack": False,
            "chop_val": 72,
            "variable": "Monthly",
            "seasonality": 12,
            "output_size": 18,
            "learning_rate": 1e-1,
            "learning_rates": ((10, 1e-4)),
            "min_learning_rate": 0.0001,
            "lr_ratio": sqrt(10),
            "lr_tolerance_multip": 1.005,
            "min_epochs_before_changing_lrate": 3,
            "lr_anneal_rate": 0.5,
            "lr_anneal_step": 5,
            "dropout": 0.2,
            # "sample_ids": [],
            "sample_ids": ["M1"],
        })
    elif interval == "Daily":
        config.update({
            "stack_types": [BLOCK_TYPE.TREND, BLOCK_TYPE.SEASONALITY],
            "thetas_dims": [2, 8],
            "nb_blocks_per_stack": 3,
            "hidden_layer_units": 256,
            "share_weights_in_stack": False,
            "variable": "Daily",
            "seasonality": 7,
            "output_size": 14,
            "learning_rate": 1e-1,
            "learning_rates": ((10, 1e-4)),
            "min_learning_rate": 0.0001,
            "lr_ratio": sqrt(10),
            "lr_tolerance_multip": 1.005,
            "min_epochs_before_changing_lrate": 2,
            "lr_anneal_rate": 0.5,
            "lr_anneal_step": 5,
            "dropout": 0.5,
            # "sample_ids": [],
            "sample_ids": ["D1"],
        })
    elif interval == "Yearly":

        config.update({
            "stack_types": [BLOCK_TYPE.TREND, BLOCK_TYPE.SEASONALITY],
            "thetas_dims": [2, 8],
            "nb_blocks_per_stack": 3,
            "hidden_layer_units": 256,
            "share_weights_in_stack": True,
            "variable": "Yearly",
            "seasonality": 1,
            "output_size": 6,
            "learning_rate": 1e-1,
            "learning_rates": ((10, 1e-4)),
            "min_learning_rate": 0.0001,
            "lr_ratio": sqrt(10),
            "lr_tolerance_multip": 1.005,
            "min_epochs_before_changing_lrate": 2,
            "lr_anneal_rate": 0.5,
            "lr_anneal_step": 5,
            "dropout": 0.2,
            #"sample_ids": [],
            "sample_ids": ["Y1"],
        })
    elif interval == "Weekly":
        config.update({
            "stack_types": [BLOCK_TYPE.TREND, BLOCK_TYPE.SEASONALITY],
            "thetas_dims": [2, 8],
            "nb_blocks_per_stack": 3,
            "hidden_layer_units": 128,
            "share_weights_in_stack": True,
            "variable": "Weekly",
            "seasonality": 1,
            "output_size": 13,
            "learning_rate": 1e-1,
            "learning_rates": ((10, 1e-4)),
            "min_learning_rate": 0.0001,
            "lr_ratio": sqrt(10),
            "lr_tolerance_multip": 1.005,
            "min_epochs_before_changing_lrate": 2,
            "lr_anneal_rate": 0.5,
            "lr_anneal_step": 5,
            "dropout": 0.2,
            # "sample_ids": [],
            "sample_ids": ["W246"],
        })
    elif interval == "Hourly":
        config.update({
            "stack_types": [BLOCK_TYPE.TREND, BLOCK_TYPE.SEASONALITY],
            "thetas_dims": [2, 8],
            "nb_blocks_per_stack": 3,
            "hidden_layer_units": 128,
            "share_weights_in_stack": False,
            "variable": "Hourly",
            "seasonality": 24,
            "output_size": 48,
            "learning_rate": 1e-1,
            "learning_rates": ((10, 1e-4)),
            "min_learning_rate": 0.0001,
            "lr_ratio": sqrt(10),
            "lr_tolerance_multip": 1.005,
            "min_epochs_before_changing_lrate": 2,
            "lr_anneal_rate": 0.5,
            "lr_anneal_step": 5,
            "dropout": 0.5,
            # "sample_ids": [],
            "sample_ids": ["H344"],
        })
    else:
        print("I dont have that config. :(")

    # config["input_size_i"] = config["input_size"]
    config["output_size_i"] = config["output_size"]
    config["tau"] = config["percentile"] / 100
    config["training_tau"] = config["training_percentile"] / 100

    if not config["prod"]:
        config["batch_size"] = 10
        config["num_of_train_epochs"] = 15

    return config

from math import sqrt

import torch

from ts.utils.helper_funcs import SAVE_LOAD_TYPE


def get_config(interval):
    config = {
        "prod": True,
        "device": ("cuda" if torch.cuda.is_available() else "cpu"),
        "percentile": 50,
        "training_percentile": 45,
        "add_nl_layer": True,
        "rnn_cell_type": "GRU",
        "num_of_train_epochs": 100,
        "num_of_train_epochs_sampling": 100,
        "num_of_categories": 6,  # in data provided
        "batch_size": 128,
        "gradient_clipping": 20,
        "sample": True,
        "reload": SAVE_LOAD_TYPE.NO_ACTION,
        "add_run_id": False,
        "save_model": SAVE_LOAD_TYPE.NO_ACTION,
        "plot_ts": True,
        "loss": ""
    }

    if interval == "Quarterly":
        config.update({
            "learning_rate": 1e-1,
            "learning_rates": ((10, 1e-4)),
            "min_learning_rate": 0.0001,
            "lr_ratio": sqrt(10),
            "lr_tolerance_multip": 1.005,
            "min_epochs_before_changing_lrate": 2,
            "lr_anneal_rate": 0.5,
            "lr_anneal_step": 5,
            "dropout": 0.2,
            "chop_val": 72,
            "variable": "Quarterly",
            "dilations": ((1, 2), (4, 8)),
            "state_hsize": 40,
            "seasonality": 4,
            "input_size": 4,
            "output_size": 8,
            "level_variability_penalty": 80,
            #"sample_ids": [],
            "sample_ids": ["Q66"],
        })
    elif interval == "Monthly":
        config.update({
            "learning_rate": 1e-2,
            "learning_rates": ((10, 1e-4)),
            "min_learning_rate": 0.0001,
            "lr_ratio": sqrt(10),
            "lr_tolerance_multip": 1.005,
            "min_epochs_before_changing_lrate": 2,
            "lr_anneal_rate": 0.5,
            "lr_anneal_step": 5,
            "chop_val": 72,
            "variable": "Monthly",
            "dilations": ((1, 3), (6, 12)),
            "state_hsize": 50,
            "seasonality": 12,
            "input_size": 12,
            "output_size": 18,
            "level_variability_penalty": 50,
            #"sample_ids": [],
             "sample_ids": ["M1"],
        })
    elif interval == "Daily":
        config.update({
            "learning_rate": 1e-1,
            "learning_rates": ((10, 1e-4)),
            "min_learning_rate": 0.0001,
            "lr_ratio": sqrt(10),
            "lr_tolerance_multip": 1.005,
            "min_epochs_before_changing_lrate": 2,
            "lr_anneal_rate": 0.5,
            "lr_anneal_step": 5,
            "chop_val": 200,
            "variable": "Daily",
            "dilations": ((1, 7), (14, 28)),
            "state_hsize": 50,
            "seasonality": 7,
            "input_size": 7,
            "output_size": 14,
            "level_variability_penalty": 50,
            # "sample_ids": [],
            "sample_ids": ["D1"],
        })
    elif interval == "Yearly":

        config.update({
            "learning_rate": 1e-1,
            "learning_rates": ((10, 1e-4)),
            "min_learning_rate": 0.0001,
            "lr_ratio": sqrt(10),
            "lr_tolerance_multip": 1.005,
            "min_epochs_before_changing_lrate": 2,
            "lr_anneal_rate": 0.5,
            "lr_anneal_step": 5,
            "chop_val": 25,
            "variable": "Yearly",
            "dilations": ((1, 2), (2, 6)),
            "state_hsize": 30,
            "seasonality": 1,
            "input_size": 4,
            "output_size": 6,
            "level_variability_penalty": 0,
            #"sample_ids": [],
            "sample_ids": ["Y1"],
        })
    elif interval == "Weekly":
        config.update({
            "learning_rate": 1e-1,
            "learning_rates": ((10, 1e-4)),
            "min_learning_rate": 0.0001,
            "lr_ratio": sqrt(10),
            "lr_tolerance_multip": 1.005,
            "min_epochs_before_changing_lrate": 2,
            "lr_anneal_rate": 0.5,
            "lr_anneal_step": 5,
            "chop_val": 72,
            "variable": "Weekly",
            "dilations": ((1, 14), (14, 28)),
            "state_hsize": 60,
            "seasonality": 1,
            "input_size": 1,
            "output_size": 13,
            "level_variability_penalty": 0,
            # "sample_ids": [],
            "sample_ids": ["W246"],
        })
    elif interval == "Hourly":
        config.update({
            "learning_rate": 1e-1,
            "learning_rates": ((10, 1e-4)),
            "min_learning_rate": 0.0001,
            "lr_ratio": sqrt(10),
            "lr_tolerance_multip": 1.005,
            "min_epochs_before_changing_lrate": 2,
            "lr_anneal_rate": 0.5,
            "lr_anneal_step": 5,
            "chop_val": 72,
            "variable": "Hourly",
            "dilations": ((1, 24), (24, 48)),
            "state_hsize": 60,
            "seasonality": 24,
            "input_size": 24,
            "output_size": 48,
            "level_variability_penalty": 0,
            #"sample_ids": [],
            "sample_ids": ["H344"],
        })
    else:
        print("I don\"t have that config. :(")

    config["input_size_i"] = config["input_size"]
    config["output_size_i"] = config["output_size"]
    config["tau"] = config["percentile"] / 100
    config["training_tau"] = config["training_percentile"] / 100

    if not config["prod"]:
        config["batch_size"] = 10
        config["num_of_train_epochs"] = 15

    return config

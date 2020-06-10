from math import sqrt

import torch

from ts.utils.helper_funcs import SAVE_LOAD_TYPE


def get_config(interval):
    config = {
        "prod": True,
        "device": ("cuda" if torch.cuda.is_available() else "cpu"),
        "num_of_train_epochs": 100,
        "num_of_train_epochs_sampling": 100,
        "num_of_categories": 6,  # in data provided
        "batch_size": 1024,
        "sample": True,
        "reload": SAVE_LOAD_TYPE.NO_ACTION,
        "add_run_id": False,
        "save_model": SAVE_LOAD_TYPE.NO_ACTION,
        "plot_ts": False,
    }

    if interval == "Quarterly":
        config.update({
            "variable": "Quarterly",
            "seasonality": 4,
            "output_size": 8,
            "learning_rate": 1e-1,
            "learning_rates": ((10, 1e-4)),
            "min_learning_rate": 0.0001,
            "lr_ratio": sqrt(10),
            "lr_tolerance_multip": 1.005,
            "min_epochs_before_changing_lrate": 2,
            "lr_anneal_rate": 0.5,
            "lr_anneal_step": 5,
            "chop_val": 72,
            "min_samples": 70,
            # "sample_ids": [],
            "sample_ids": ["Q66"],
        })
    elif interval == "Monthly":
        config.update({
            "variable": "Monthly",
            "seasonality": 12,
            "output_size": 18,
            "learning_rate": 1e-1,
            "learning_rates": ((10, 1e-4)),
            "min_learning_rate": 0.0001,
            "lr_ratio": sqrt(10),
            "lr_tolerance_multip": 1.005,
            "min_epochs_before_changing_lrate": 2,
            # "sample_ids": [],
            "lr_anneal_rate": 0.5,
            "lr_anneal_step": 5,
            "chop_val": 72,
            "min_samples": 70,
            "sample_ids": ["M1"],
        })
    elif interval == "Daily":
        config.update({
            "variable": "Daily",
            "seasonality": 7,
            "output_size": 14,
            "learning_rate": 1e-1,
            "learning_rates": ((10, 1e-4)),
            "min_learning_rate": 0.0001,
            "lr_ratio": sqrt(10),
            "lr_tolerance_multip": 1.005,
            "min_epochs_before_changing_lrate": 2,
            "chop_val": 200,
            "min_samples": 200,
            # "sample_ids": [],
            "sample_ids": ["D1"],
        })
    elif interval == "Yearly":

        config.update({
            "variable": "Yearly",
            "seasonality": 1,
            "output_size": 6,
            "learning_rate": 1e-1,
            "learning_rates": ((10, 1e-4)),
            "min_learning_rate": 0.0001,
            "lr_ratio": sqrt(10),
            "lr_tolerance_multip": 1.005,
            "min_epochs_before_changing_lrate": 2,
            "chop_val": 25,
            "min_samples": 20,
            # "sample_ids": [],
            "sample_ids": ["Y1"],
        })
    elif interval == "Weekly":
        config.update({
            "variable": "Weekly",
            "seasonality": 1,
            "output_size": 13,
            "learning_rate": 1e-1,
            "learning_rates": ((10, 1e-4)),
            "min_learning_rate": 0.0001,
            "lr_ratio": sqrt(10),
            "lr_tolerance_multip": 1.005,
            "min_epochs_before_changing_lrate": 2,
            "chop_val": 72,
            "min_samples": 72,
            # "sample_ids": [],
            "sample_ids": ["W246"],
        })
    elif interval == "Hourly":
        config.update({
            "variable": "Hourly",
            "seasonality": 24,
            "output_size": 48,
            "learning_rate": 1e-1,
            "learning_rates": ((10, 1e-4)),
            "min_learning_rate": 0.0001,
            "lr_ratio": sqrt(10),
            "lr_tolerance_multip": 1.005,
            "min_epochs_before_changing_lrate": 2,
            "chop_val": 72,
            "min_samples": 72,
            # "sample_ids": [],
            "sample_ids": ["H344"],
        })
    else:
        print("I dont have that config. :(")

    if not config["prod"]:
        config["batch_size"] = 10
        config["num_of_train_epochs"] = 15

    return config

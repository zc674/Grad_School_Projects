import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ts.abstract_trainer import BaseTrainer
from ts.utils.helper_funcs import plot_ts
from ts.utils.loss_modules import np_sMAPE, np_MASE, np_mase


class Trainer(BaseTrainer):
    def __init__(self, model_name, model, optimizer, criterion, dataloader, run_id, add_run_id, config, forecast_length,
                 backcast_length,
                 ohe_headers,
                 csv_path, figure_path, sampling, reload):
        super().__init__(model_name, model, optimizer, criterion, dataloader, run_id, add_run_id, config, ohe_headers,
                         csv_path, figure_path, sampling, reload)
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length

    def train_batch(self, train, val, test, info_cat, idx):
        self.optimizer.zero_grad()

        window_input_list = []
        window_output_list = []
        ts_len = train.shape[1]
        for i in range(self.backcast_length, ts_len - self.forecast_length):
            window_input_list.append(train[:, i - self.backcast_length:i])
            window_output_list.append(train[:, i:i + self.forecast_length])

        if len(window_output_list) == 1:
            window_input = torch.cat([i.unsqueeze(1) for i in window_input_list], dim=0)
            window_output = torch.cat([i.unsqueeze(1) for i in window_output_list], dim=0)
        else:
            window_input = torch.cat([i.unsqueeze(0) for i in window_input_list], dim=0)
            window_input = window_input.transpose(0, 1)
            window_output = torch.cat([i.unsqueeze(0) for i in window_output_list], dim=0)
            window_output = window_output.transpose(0, 1)

        backcast, forecast = self.model(window_input)
        loss = self.criterion(forecast, window_output)
        loss.backward()
        nn.utils.clip_grad_value_(self.model.parameters(), self.config["gradient_clipping"])
        self.optimizer.step()
        return float(loss)

    def val(self, file_path, testing, debugging, figure_path):
        self.model.eval()
        with torch.no_grad():
            acts = []
            preds = []
            total_acts = []
            info_cats = []
            hold_out_loss = 0
            for batch_num, (train, val, test, info_cat, _, idx) in enumerate(self.data_loader):
                target = test if testing else val
                if testing:
                    train = torch.cat((train, val), dim=1)
                ts_len = train.shape[1]
                input = train[:, ts_len - self.backcast_length:ts_len][np.newaxis, ...]
                backcast, forecast = self.model(input)
                hold_out_loss += self.criterion(forecast, target[np.newaxis, ...])
                acts.extend(target.view(-1).cpu().detach().numpy())
                preds.extend(forecast.view(-1).cpu().detach().numpy())
                total_act = torch.cat((train, forecast.view(forecast.shape[1], -1)), dim=1)
                total_acts.extend(total_act.view(-1).cpu().detach().numpy())
                info_cats.append(info_cat.cpu().detach().numpy())

            hold_out_loss = hold_out_loss / (batch_num + 1)

            info_cat_overall = np.concatenate(info_cats, axis=0)
            _hold_out_df = pd.DataFrame({"acts": acts, "preds": preds})
            cats = [val for val in self.ohe_headers[info_cat_overall.argmax(axis=1)] for _ in
                    range(self.config["output_size"])]
            _hold_out_df["category"] = cats

            overall_hold_out_df = copy.copy(_hold_out_df)
            overall_hold_out_df["category"] = ["Overall" for _ in cats]

            overall_hold_out_df = pd.concat((_hold_out_df, overall_hold_out_df), sort=False)

            mase = np_mase(total_acts, self.config["output_size"])
            grouped_results = overall_hold_out_df.groupby(["category"]).apply(
                lambda x: np_MASE(x.preds, x.acts, mase, x.shape[0]))
            results = grouped_results.to_dict()
            print("============== MASE ==============")
            print(results)

            grouped_results = overall_hold_out_df.groupby(["category"]).apply(
                lambda x: np_sMAPE(x.preds, x.acts, x.shape[0]))
            results = grouped_results.to_dict()

            print("============== sMAPE ==============")
            print(results)

            hold_out_loss = float(hold_out_loss.detach().cpu())
            print("============== HOLD-OUT-LOSS ==============")
            print("hold_out_loss:{:5.2f}".format(hold_out_loss))

            results["hold_out_loss"] = hold_out_loss
            self.log_values(results)

            grouped_path = file_path / ("grouped_results-{}.csv".format(self.epochs))
            grouped_results.to_csv(grouped_path, header=True)

        return hold_out_loss

    def plot(self, testing=False):
        self.model.eval()
        with torch.no_grad():
            (train, val, test, info_cat, ts_labels, idx) = next(iter(self.data_loader))
            target = test if testing else val
            info_cats = info_cat.cpu().detach().numpy()
            cats = [val for val in self.ohe_headers[info_cats.argmax(axis=1)]]

            if testing:
                train = torch.cat((train, val), dim=1)
            ts_len = train.shape[1]
            input = train[:, ts_len - self.backcast_length:ts_len][np.newaxis, ...]
            backcast, forecast = self.model(input)
            original_ts = torch.cat((train, target), axis=1)
            predicted_ts = torch.cat((train, forecast.squeeze(axis=0)), axis=1)
            plot_ts(original_ts, predicted_ts, ts_labels, cats, self.figure_path, number_to_plot=train.shape[0])

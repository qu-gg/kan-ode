"""
@file CommonMetaDynamics.py

A common class that each meta latent dynamics function inherits.
Holds the training + validation step logic and the VAE components for reconstructions.
Has a testing step for holdout steps that handles all metric calculations and visualizations.
"""
import os
import json
import torch
import numpy as np
import pytorch_lightning
import matplotlib.pyplot as plt

from utils import metrics
from utils.utils import MAELoss
from utils.plotting import show_sequences


class LatentDynamicsModel(pytorch_lightning.LightningModule):
    def __init__(self, args):
        """ Generic training and testing boilerplate for the dynamics models """
        super().__init__()
        self.save_hyperparameters(args)

        # Args
        self.args = args

        # Losses
        self.reconstruction_loss = MAELoss()

        # General trackers
        self.n_updates = 0

        # List to hold per-batch outputs
        self.outputs = list()

    def forward(self, x, controls, generation_training_len):
        """ Placeholder function for the dynamics forward pass """
        raise NotImplementedError("In forward: Latent Dynamics function not specified.")

    def model_specific_loss(self, x, domain, preds, train=True):
        """ Placeholder function for any additional loss terms a dynamics function may have """
        return 0.0

    def model_specific_plotting(self, version_path, outputs):
        """ Placeholder function for any additional plots a dynamics function may have """
        return None

    @staticmethod
    def get_model_specific_args():
        """ Placeholder function for model-specific arguments """
        return {}

    def configure_optimizers(self):
        """ By default, we assume a optim with the Adam Optimizer """
        optim = torch.optim.SGD(self.parameters(), lr=self.args.learning_rate)
        return optim

    def on_train_start(self):
        """ Boilerplate experiment logging setup pre-training """
        # Get total number of parameters for the model and save
        self.log("total_num_parameters", float(sum(p.numel() for p in self.parameters() if p.requires_grad)), prog_bar=False)

        # Make image dir in lightning experiment folder if it doesn't exist
        if not os.path.exists(f"{self.logger.log_dir}/signals/"):
            os.mkdir(f"{self.logger.log_dir}/signals/")

    def get_step_outputs(self, batch, generation_len):
        """
        Handles the process of pre-processing and subsequence sampling a batch,
        as well as getting the outputs from the models regardless of step
        :param batch: list of dictionary objects representing a single image
        :param generation_training_len: how far out to generate for, dependent on the step (train/val)
        :return: processed model outputs
        """
        # Stack batch and restrict to generation length
        indices, signals, controls = batch

        # Get predictions
        preds = self(signals, controls, generation_len)
        return signals, controls, preds

    def get_step_losses(self, signals, domains, preds, is_train=True):
        """
        Handles getting the ELBO terms for the given step
        :param signals: ground truth signals
        :param preds: forward predictions from the model
        :return: likelihood, kl on z0, model-specific dynamics loss
        """
        # Reconstruction loss for the sequence and z0
        likelihood = self.reconstruction_loss(preds, signals).mean([1, 2]).mean()
        init_likelihood = 0

        # Get the loss terms from the specific latent dynamics loss
        dynamics_loss = self.model_specific_loss(signals, domains, preds, is_train)
        return likelihood, init_likelihood, dynamics_loss

    def get_metrics(self, outputs, setting):
        """ Takes the dictionary of saved batch metrics, stacks them, and gets outputs to log in the Tensorboard """
        # Convert outputs to Tensors and then Numpy arrays
        signals = torch.vstack([out["signals"] for out in outputs])
        preds = torch.vstack([out["preds"] for out in outputs])

        # Iterate through each metric function and add to a dictionary
        out_metrics = {}
        for met in self.args.metrics:
            metric_function = getattr(metrics, met)
            out_metrics[met] = metric_function(signals, preds, args=self.args)[1]

        # Return a dictionary of metrics
        return out_metrics

    def training_step(self, batch, batch_idx):
        """
        PyTorch-Lightning training step where the network is propagated and returns a single loss value,
        which is automatically handled for the backward update
        :param batch: list of dictionary objects representing a single image
        :param batch_idx: how far in the epoch this batch is
        """
        # Get model outputs from batch
        signals, controls, preds = self.get_step_outputs(batch, self.args.timesteps)

        # Get model loss terms for the step
        likelihood, init_likelihood, dynamics_loss = self.get_step_losses(signals, controls, preds, is_train=True)

        # Build the full loss
        loss = likelihood + 5 * init_likelihood + dynamics_loss

        # Log loss terms
        self.log_dict(
            {"likelihood": likelihood, "init_likelihood": 5 * init_likelihood,"dynamics_loss": dynamics_loss},
            prog_bar=True
        )

        # Return outputs as dict
        self.outputs.append({"loss": loss, "preds": preds.detach(), "signals": signals.detach(), "controls": controls.detach()})
        self.n_updates += 1
        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Log metrics over saved batches on the specified interval
        if batch_idx % self.args.log_interval == 0 and batch_idx != 0:
            metrics = self.get_metrics(self.outputs[:50], setting='train')
            for metric in metrics.keys():
                self.log(f"train_{metric}", metrics[metric], prog_bar=True)

            # Show side-by-side reconstructions
            show_sequences(self.outputs[0]["signals"], self.outputs[0]["preds"],
                           f'{self.logger.log_dir}/signals/recon{self.n_updates}train.png', num_out=4)

            # Model-specific plots
            self.model_specific_plotting(self.logger.log_dir, self.outputs[:50])

            # Wipe the saved batch outputs
            self.outputs = list()

    def test_step(self, batch, batch_idx):
        # Get model outputs from batch
        signals, controls, preds = self.get_step_outputs(batch, self.args.timesteps)

        # Return outputs as dict
        out = {"preds": preds.detach().cpu(), "signals": signals.detach().cpu(), "controls": controls.detach().cpu()}
        return out

    def test_epoch_end(self, batch_outputs):
        """ For testing end, save the predictions, gt, and MSE to NPY files in the respective experiment folder """
        # Set up output path and create dir
        output_path = f"{self.logger.log_dir}/test"
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # Stack all output types and convert to numpy
        outputs = dict()
        for key in batch_outputs[0].keys():
            outputs[key] = torch.vstack([output[key] for output in batch_outputs])

        # Save to files
        if self.args.save_files is True:
            for key in outputs.keys():
                np.save(f"{output_path}/test_{key}.npy", outputs[key].numpy())

        # Iterate through each metric function and add to a dictionary
        print("\n=> getting metrics...")
        out_metrics = {}
        for met in self.args.metrics:
            metric_function = getattr(metrics, met)
            metric_results, metric_mean, metric_std = metric_function(outputs["signals"], outputs["preds"], args=self.args, setting='test')
            out_metrics[f"{met}_mean"], out_metrics[f"{met}_std"] = float(metric_mean), float(metric_std)
            print(f"=> {met}: {metric_mean:4.5f}+-{metric_std:4.5f}")
            np.save(f"{output_path}/test_{met}.npy", metric_results)

        # Save metrics to JSON in checkpoint folder
        with open(f"{output_path}/test_metrics.json", 'w') as f:
            json.dump(out_metrics, f)

        # Show side-by-side reconstructions
        show_sequences(outputs["signals"][:10], outputs["preds"][:10], f"{output_path}/test_examples.png", num_out=5)

        # Plot the reconstructed full mesh
        plt.plot(range(20), outputs["signals"][:, :, 0].detach().cpu().numpy().T, c='b')
        plt.plot(range(20), outputs["preds"][:, :, 0].detach().cpu().numpy().T, c='k', linestyle='--')
        plt.title(f"Plot of U[-2, 2]")
        plt.xlabel("Blue: GT | Black: Preds")

        plt.savefig(f"{output_path}/reconstructedControls.png")
        plt.close()

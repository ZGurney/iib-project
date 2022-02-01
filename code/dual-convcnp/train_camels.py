import argparse
import os
import joblib

import lab.torch as B
import matplotlib.pyplot as plt
import numpy as np
import torch
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak
import pickle

import convcnp
import data_generator

# Enable GPU if it is available.
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Open file with pickle
file_path = "data_generator/maurer.pickle"
file = open(file_path, "rb")
data = pickle.load(file)
file.close()

HydroGenerator = data_generator.HydroGenerator(
    dict_df = data,
    channels_c = ["PRCP(mm/day)", "Q"],
    channels_t = ["PRCP(mm/day)", "Q"],
)

def threshold_data(data, threshold_value):
    formatted_data = torch.unsqueeze(data[:, :, 0], 2)
    thresholded_data = (B.sign(formatted_data - threshold_value*torch.ones_like(formatted_data)) + 1) / 2
    return thresholded_data

def generate_batches(num_batches):
    """Generate batches from the CAMELS dataset"""
    
    batches = []
    for i in range(num_batches):
        batch = HydroGenerator.generate_task()
        batch = {
            "x": batch["x"],
            "y": batch["y"],
            "x_context_class": batch["x_context"],
            "asds": (B.sign(tensor - torch.ones_like(tensor)) + 1) / 2
            "y_context_class": torch.unsqueeze(batch["y_context"][:, :, 1], 2),
            "x_target_class": batch["x_target"],
            "y_target_class": torch.unsqueeze(batch["y_target"][:, :, 1], 2),
            "x_context_reg": batch["x_context"],
            "y_context_reg": torch.unsqueeze(batch["y_context"][:, :, 0], 2),
            "x_target_reg": batch["x_target"],
            "y_target_reg": torch.unsqueeze(batch["y_target"][:, :, 0], 2),
        }
        batches.append(batch)
    return batches


def compute_loss(model, batch, mode="dual"):
    """Compute the sum of the classification and regression loss functions."""
    
    def LogSigmoid(x):
        """Compute the log-sigmoid element-wise, without numerical issues"""
        c = B.max(-x, 0)
        return -c - B.log(B.exp(-c) + B.exp(-x - c))
        #return B.log(B.sigmoid(x))

    class_prob, (reg_mean, reg_std) = model(batch)

    if mode == "dual":

        # Clamp the classification probabilities to prevent the loss for NaNing out.
        #class_prob = class_prob.clamp(1e-4, 1 - 1e-4)

        class_loss = -B.sum(
            batch["y_target_class"] * LogSigmoid(class_prob)
            + (1 - batch["y_target_class"]) * LogSigmoid(1 - class_prob)
        )

        reg_loss = 0.5 * B.sum(
            B.log_2_pi
            + B.log(reg_std)
            + ((reg_mean - batch["y_target_reg"]) / reg_std) ** 2
        )

        overall_loss = args.alpha * class_loss + (1 - args.alpha) * reg_loss

    if mode == "classification":

        # Clamp the classification probabilities to prevent the loss for NaNing out.
        #class_prob = class_prob.clamp(1e-4, 1 - 1e-4)

        overall_loss = -B.sum(
            batch["y_target_class"] * LogSigmoid(class_prob)
            + (1 - batch["y_target_class"]) * LogSigmoid(-class_prob)
        )

    if mode == "regression":

        overall_loss = 0.5 * B.sum(
            B.log_2_pi
            + B.log(reg_std)
            + ((reg_mean - batch["y_target_reg"]) / reg_std) ** 2
        )

    return overall_loss


def take_first(x, convert_to_numpy=True):
    """Take the first of a batch."""
    if B.rank(x) > 1:
        x = x[0, :, 0]
        if convert_to_numpy:
            x = B.to_numpy(x)
    return x

# Plotting script
def plot_graphs(batch, epoch, n):
    """
    Plot classification and regression graphs for different proportions 
    of classification and regression context points
    """

    # Set up batch to compute loss on single task
    task = {}
    for key, value in batch.items():
        task[key] = take_first(value, convert_to_numpy=False)
    if not mode == "regression":
        class_loss = compute_loss(model, task, mode="classification")
        print(f"Classification loss {n}: {class_loss:6.2f}")
    if not mode == "classification":
        reg_loss = compute_loss(model, task, mode="regression")
        print(f"Regression loss {n}: {reg_loss:6.2f}")


    with B.on_device(device):
        # Set `x_target` to a dense linspace for the plots, but save the original
        # target inputs.
        x_target_class = batch["x_target_class"]
        x_target_reg = batch["x_target_reg"]
        batch["x_target_class"] = B.linspace(torch.float32, *HydroGenerator.x_range, 200)
        batch["x_target_reg"] = B.linspace(torch.float32, *HydroGenerator.x_range, 200)

        class_prob, (reg_mean, reg_std) = model(batch)
        class_prob = B.sigmoid(class_prob)
    

    # Plot for classification:

    if not mode == "regression":
        plt.figure()
        plt.title(f"Classification (Epoch {epoch + 1})")
        plt.scatter(
            take_first(batch["x_context_class"]),
            take_first(batch["y_context_class"]),
            style="train",
            label="Context",
        )
        plt.scatter(
            take_first(x_target_class),
            take_first(batch["y_target_class"]),
            style="test",
            label="Target",
        )
        plt.plot(
            take_first(batch["x_target_class"]),
            take_first(class_prob),
            style="pred",
            label="Prediction",
        )
        tweak(legend_loc="best")
        plt.savefig(wd.file(f"epoch{epoch + 1}_classification{n}.pdf"))
        plt.close()

    # Plot for regression:

    if not mode == "classification":
        plt.figure()
        plt.title(f"Regression (Epoch {epoch + 1})")
        plt.scatter(
            take_first(batch["x_context_reg"]),
            take_first(batch["y_context_reg"]),
            style="train",
            label="Context",
        )
        plt.scatter(
            take_first(x_target_reg),
            take_first(batch["y_target_reg"]),
            style="test",
            label="Target",
        )
        plt.plot(
            take_first(batch["x_target_reg"]),
            take_first(reg_mean),
            style="pred",
            label="Prediction",
        )
        plt.fill_between(
            take_first(batch["x_target_reg"]),
            take_first(reg_mean - 1.96 * reg_std),
            take_first(reg_mean + 1.96 * reg_std),
            style="pred",
        )
        tweak(legend_loc="best")
        plt.savefig(wd.file(f"epoch{epoch + 1}_regression{n}.pdf"))
        plt.close()

# Evaluation script
def evaluate_model(model, mode, epoch):
    losses = []
    i = 0.2
    for batch in batches_test:
        losses.append(compute_loss(model, batch, mode))
        i += 0.2
    losses = B.to_numpy(losses)
    error = 1.96 * np.std(losses) / np.sqrt(len(losses))
    print(f"Overall loss: {np.mean(losses):6.2f} +- {error:6.2f}")

    # Produce some plots.
    print("Plotting...")
    plot_graphs(batches_test[0], epoch, n=1)

    # Save checkpoint
    checkpoint = {
        "epoch": epoch + 1,
        "loss": (np.mean(losses), error),
        "state_dict": model.state_dict(),
        "optimiser": opt.state_dict(),
    }

    # Save the trained model in the outputs folder
    model_file_name = f"model{epoch + 1}.tar"
    with open(model_file_name, "wb") as file:
        joblib.dump(value=checkpoint, filename=os.path.join('./outputs/', model_file_name))

# Parse command line arguments.
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--root",
    type=str,
    default="outputs",
    help="Directory to store output of experiment.",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="Number of epochs to run for.",
)
parser.add_argument(
    "--tasks_per_epoch",
    type=int,
    default=16_384,
    help="Number of tasks per epoch.",
)
parser.add_argument(
    "--small",
    action="store_true",
    help="Use a small CNN architecture.",
)
parser.add_argument(
    "--rate",
    type=float,
    default=5e-3,
    help="Learning rate.",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.5,
    help="Weight assigned to the classification loss.",
)
parser.add_argument(
    "--mode",
    choices=["dual", "classification", "regression"],
    default="dual",
    help="Mode of operation (dual, classification, regression)",
)
args = parser.parse_args()

# Setup working directory.
wd = WorkingDirectory(args.root, seed=0, override=True)

# Setup data generator.
batches_train = generate_batches(num_batches=2**8)
batches_test = generate_batches(num_batches=4)

# Construct model.
mode = args.mode
if mode == "dual":
    model = convcnp.DualConvCNP(small=args.small).to(device)
if mode == "classification":
    model = convcnp.ClassConvCNP(small=args.small).to(device)
if mode == "regression":
    model = convcnp.RegConvCNP(small=args.small).to(device)

# Construct optimiser.
opt = torch.optim.Adam(params=model.parameters(), lr=args.rate)

# Compute eval loss for epoch 0 (no training)
print("Evaluating epoch 0 (before training)")
evaluate_model(model, mode, epoch=-1)

# Run training loop.
for epoch in range(args.epochs):
    print(f"Starting epoch {epoch + 1}")

    # Run training epoch.
    print("Training...")
    for batch in batches_train:
        loss = compute_loss(model, batch, mode)
        # Perform gradient step.
        loss.backward()
        opt.step()
        opt.zero_grad()
    
    # Compute eval loss and save model.
    print("Evaluating...")
    evaluate_model(model, mode, epoch)
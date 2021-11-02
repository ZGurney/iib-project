import argparse
import os
import joblib

import lab.torch as B
import matplotlib.pyplot as plt
import numpy as np
import torch
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak

from convcnp import DualConvCNP, ClassConvCNP, RegConvCNP, GPGenerator

from azureml.core import Run # Import library for logging in Azure

run = Run.get_context() # Access run object for logging

evaluation_tasks = torch.load("evaluation-tasks.pt")

# Enable GPU if it is available.
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def split_off_classification(batch, proportion_class="random"):
    """Split off a classification data set."""
    n_context = B.shape(batch["x_context"], 1)

    # Set ratio of classification to regression context datapoints
    if proportion_class == "random":
        n_class = np.random.randint(low=1, high=n_context - 1)
    else:
        n_class = int(proportion_class*n_context)

    return {
        "x_context_class": batch["x_context"][:, :n_class, :],
        "y_context_class": (B.sign(batch["y_context"][:, :n_class, :]) + 1) / 2,
        "x_target_class": batch["x_target"][:, :n_class, :],
        "y_target_class": (B.sign(batch["y_target"][:, :n_class, :]) + 1) / 2,
        "x_context_reg": batch["x_context"][:, n_class:, :],
        "y_context_reg": batch["y_context"][:, n_class:, :],
        "x_target_reg": batch["x_target"][:, n_class:, :],
        "y_target_reg": batch["y_target"][:, n_class:, :],
    }


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


# Parse command line arguments.
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--root",
    type=str,
    default="_experiments/experiment",
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
    type=str,
    default="dual",
    help="Mode of operation (dual, classification, regression)",
)
args = parser.parse_args()

# Setup working directory.
wd = WorkingDirectory(args.root, seed=0, override=True)

# Setup data generator.
gen_train = GPGenerator(num_tasks=args.tasks_per_epoch)
gen_test = GPGenerator(num_tasks=64)

# Construct model.
mode = args.mode
if mode == "dual":
    model = DualConvCNP(small=args.small).to(device)
if mode == "classification":
    model = ClassConvCNP(small=args.small).to(device)
if mode == "regression":
    model = RegConvCNP(small=args.small).to(device)

# Construct optimiser.
opt = torch.optim.Adam(params=model.parameters(), lr=args.rate)

# Plotting script
def plot_graphs(batch, proportion_class, n):
    """
    Plot classification and regression graphs for different proportions 
    of classification and regression context points
    """
    batch = split_off_classification(batch, proportion_class)

    with B.on_device(device):
        # Set `x_target` to a dense linspace for the plots, but save the original
        # target inputs.
        x_target_class = batch["x_target_class"]
        x_target_reg = batch["x_target_reg"]
        batch["x_target_class"] = B.linspace(torch.float32, *gen_test.x_range, 200)
        batch["x_target_reg"] = B.linspace(torch.float32, *gen_test.x_range, 200)

        class_prob, (reg_mean, reg_std) = model(batch)
        class_prob = B.sigmoid(class_prob)

    for key in batch.keys():
        batch[key] = take_first(batch[key], convert_to_numpy=False)

    # Plot for classification:

    class_loss = compute_loss(model, batch, mode="classification")
    print(f"Classification loss {n}: {class_loss:6.2f}")

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
    plt.savefig(wd.file(f"./outputs/epoch{epoch + 1}_classification{n}.pdf"))
    run.log_image(name=f"epoch{epoch + 1}_classification{n}", plot=plt)
    plt.close()

    # Plot for regression:

    reg_loss = compute_loss(model, batch, mode="regression")
    print(f"Regression loss {n}: {reg_loss:6.2f}")

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
    plt.savefig(wd.file(f"./outputs/epoch{epoch + 1}_regression{n}.pdf"))
    run.log_image(name=f"epoch{epoch + 1}_regression{n}", plot=plt)
    plt.close()

# Evaluation script
def evaluate_model(model, mode):
    with torch.no_grad():
        losses = []
        for batch in evaluation_tasks:
            batch = split_off_classification(batch)
            losses.append(compute_loss(model, batch, mode))
        losses = B.to_numpy(losses)
        error = 1.96 * np.std(losses) / np.sqrt(len(losses))
        print(f"Overall loss: {np.mean(losses):6.2f} +- {error:6.2f}")
        run.log(name="loss", value=np.mean(losses))
        run.log(name="error", value=error)

        # Produce some plots.
        print("Plotting...")
        
        # Sparse classification data
        plot_graphs(evaluation_tasks[0], proportion_class=0.2, n=1)
        # Sparse regression data
        plot_graphs(evaluation_tasks[2], proportion_class=0.8, n=2)
        # Basic example
        plot_graphs(evaluation_tasks[3], proportion_class=0.5, n=3)

        # Save checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "loss": (np.mean(losses), error),
            "state_dict": model.state_dict(),
            "optimiser": opt.state_dict(),
        }

        # !!!Check if unecessary!!!
        # Check if output folder already made
        if not os.path.isdir("./outputs"):
            print("Output folder not yet made, initialise folder")
            os.makedirs('./outputs', exist_ok=True)

        # Save the trained model in the outputs folder
        model_file_name = f"model{epoch + 1}.tar"
        with open(model_file_name, "wb") as file:
            joblib.dump(value=checkpoint, filename=os.path.join('./outputs/', model_file_name))

# Run training loop.
for epoch in range(args.epochs):
    print(f"Starting epoch {epoch + 1}")

    # Run training epoch.
    print("Training...")
    for batch in gen_train.epoch(device):
        batch = split_off_classification(batch)
        loss = compute_loss(model, batch, mode)
        # Perform gradient step.
        loss.backward()
        opt.step()
        opt.zero_grad()
    
    # Compute eval loss and save model.
    print("Evaluating...")
    evaluate_model(model, mode)

run.complete()
import argparse
import os
import joblib

import lab.torch as B
import matplotlib.pyplot as plt
import numpy as np
import torch
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak

import convcnp

# Enable GPU if it is available.
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Load evaluation tasks already pre-generated.
evaluation_tasks = torch.load("evaluation-tasks.pt", map_location=torch.device(device))

def split_off_classification(batch, proportion_class="random"):
    """Split off a classification data set."""
    n_context = B.shape(batch["x_context"], 1)

    # Set ratio of classification to regression context datapoints
    if proportion_class == "random":
        n_class = np.random.randint(low=1, high=n_context - 1)
    else:
        n_class = int(proportion_class*n_context)

    return {
        "x": batch["x"],
        "y": batch["y"],
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

# Plotting script
def plot_graphs(batch, epoch, proportion_class, n):
    """
    Plot classification and regression graphs for different proportions 
    of classification and regression context points
    """
    batch = split_off_classification(batch, proportion_class)

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
        batch["x_target_class"] = B.linspace(torch.float32, *gen_test.x_range, 200)
        batch["x_target_reg"] = B.linspace(torch.float32, *gen_test.x_range, 200)

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
    for batch in evaluation_tasks:
        batch = split_off_classification(batch, proportion_class=i)
        losses.append(compute_loss(model, batch, mode))
        i += 0.2
    losses = B.to_numpy(losses)
    error = 1.96 * np.std(losses) / np.sqrt(len(losses))
    print(f"Overall loss: {np.mean(losses):6.2f} +- {error:6.2f}")

    # Produce some plots.
    print("Plotting...")
    
    kernel = stheno.EQ().stretch(0.25)
    seed = 2
    plot_graphs(generate_task(kernel, seed), epoch, 0.5, n=1)

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

def generate_task(kernel, seed):
    """
    Generates a single task with specified seed
    """
    gen_test_comparison = convcnp.GPGenerator2(kernel=kernel, noise=0, seed=seed, batch_size=1, num_tasks=1, num_context_points=50)
    test_comparison = gen_test_comparison.generate_batch()

    # Sort both the context and target sets
    test_comparison["y_context"] = test_comparison["y_context"][:, B.flatten(B.argsort(test_comparison["x_context"], axis=1)), :]
    test_comparison["x_context"] = B.sort(test_comparison["x_context"], axis=1)
    test_comparison["y_target"] = test_comparison["y_target"][:, B.flatten(B.argsort(test_comparison["x_target"], axis=1, descending=True)), :]
    test_comparison["x_target"] = B.sort(test_comparison["x_target"], axis=1, descending=True)

    return test_comparison

# Parse command line arguments.
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--data",
    choices=['eq',
             'matern',
             'noisy-mixture',
             'weakly-periodic',
             'sawtooth'],
    default="eq",
    help='Data set to train the CNP on. '
)
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
gen_train = convcnp.GPGenerator(num_tasks=args.tasks_per_epoch)
gen_test = convcnp.GPGenerator(num_tasks=64)

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
    for batch in gen_train.epoch(device):
        batch = split_off_classification(batch)
        loss = compute_loss(model, batch, mode)
        # Perform gradient step.
        loss.backward()
        opt.step()
        opt.zero_grad()
    
    # Compute eval loss and save model.
    print("Evaluating...")
    evaluate_model(model, mode, epoch)
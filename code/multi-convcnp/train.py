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

def LogSigmoid(x):
        """Compute the log-sigmoid element-wise, without numerical issues"""
        c = B.max(-x, 0)
        return -c - B.log(B.exp(-c) + B.exp(-x - c))

def compute_loss(model, batch):
    """Compute the mean of all classification and regression loss functions."""

    predictions = model(batch)

    # Count how many classification outputs are present in batch
    n_class = 0
    for output in batch:
        if output["type"] == "classification":
            n_class += 1

    losses = []

    for output_index, class_prob in enumerate(predictions[:n_class]):
        assert batch[output_index]["type"] == "classification", f"Output {output_index} is not classification type"

        losses.append(-B.sum(
            batch[output_index]["y_target"] * LogSigmoid(class_prob)
            + (1 - batch[output_index]["y_target"]) * LogSigmoid(-class_prob)
        ))

    for output_index, (reg_mean, reg_std) in enumerate(predictions[n_class:]):
        assert batch[output_index]["type"] == "regression", f"Output {output_index} is not regression type"

        losses.append(0.5 * B.sum(
            B.log_2_pi
            + B.log(reg_std)
            + ((reg_mean - batch["y_target"]) / reg_std) ** 2
        ))

    assert len(losses) == len(batch), f"Size of loss vector is not {len(batch)}"

    overall_loss = B.mean(losses)

    return overall_loss


def take_first(x, convert_to_numpy=True):
    """Take the first of a batch."""
    if B.rank(x) > 1:
        x = x[0, :, 0]
        if convert_to_numpy:
            x = B.to_numpy(x)
    return x

# Plotting script
def plot_graphs(batch, epoch, i):
    """
    Plot classification and regression graphs for different proportions 
    of classification and regression context points
    Batch is of form: 
    [ output 1
         …
      output N ]

    with each output:
    { "type": "classification" or "regresssion"
      "x_context": [ … ]
      "y_context": [ … ]
      "x_target": [ … ]
      "y_target": [ … ]
    }
    """

    for output_index, output in enumerate(batch):

        # Set up output to compute loss on single task
        single_task = {}
        for key, value in output.items():
            single_task[key] = take_first(value, convert_to_numpy=False)

        # NOTE: add square brackets to make it look like a batch with multiple outputs?
        loss = compute_loss(model, [single_task])
        print(f"Output {output_index} loss: {loss:6.2f}")


        with B.on_device(device):
            # Set `x_target` to a dense linspace for the plots, but save the original
            # target inputs.
            """x_target_class = batch["x_target_class"]
            x_target_reg = batch["x_target_reg"]
            batch["x_target_class"] = B.linspace(torch.float32, *gen_test.x_range, 200)
            batch["x_target_reg"] = B.linspace(torch.float32, *gen_test.x_range, 200)"""
            # NOTE: how do I set the x range?           
            x_range = (B.min(), B.max())
            x_dense = B.linspace(torch.float32, *x_range, 200)

            # NOTE: indexing at output index to extract single prediction
            prediction = model(batch)[output_index]
            if output["type"] == "classification":
                prediction = B.sigmoid(prediction)

        for key, value in output.items():
            single_task[key] = take_first(value)

        plt.figure()
        plt.title(f"Output {output_index} (Epoch {epoch+1})")
        plt.scatter(
            output["x_context"],
            output["y_context"],
            style="train",
            label="Context",
        )
        plt.scatter(
            output["x_target"],
            output["x_target"],
            style="test",
            label="Target",
        )
        if output["type"] = "classification":
            plt.plot(
                x_dense,
                take_first(prediction)
                style="pred",
                label="Prediction",
            )
        else:
            reg_mean, reg_std = prediction
            plt.plot(
                x_dense,
                take_first(prediction),
                style="pred",
                label="Prediction",
            )
            plt.fill_between(
                x_dense,
                take_first(reg_mean - 1.96 * reg_std),
                take_first(reg_mean + 1.96 * reg_std),
                style="pred",
            )
        tweak(legend_loc="best")
        plt.savefig(wd.file(f"epoch{epoch+1}_{i}_output{output_index}.pdf"))
        plt.close()

# Evaluation script
def evaluate_model(model, epoch, evaluation_tasks):
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
    plot_graphs(generate_task(kernel, seed), epoch, i=1)

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
    "--output_structure",
    type=list,
    default=[1,1]
    help="Number of discrete binary outputs and continuous outputs",
)
args = parser.parse_args()

# Setup working directory.
wd = WorkingDirectory(args.root, seed=0, override=True)

# Setup data generator.
data_train = convcnp.DataGenerator(num_tasks=args.tasks_per_epoch)
data_test = convcnp.DataGenerator(num_tasks=64)

# Construct model.
model = convcnp.MultiConvCNP(small=args.small).to(device)

# Construct optimiser.
opt = torch.optim.Adam(params=model.parameters(), lr=args.rate)

# Compute eval loss for epoch 0 (no training)
print("Evaluating epoch 0 (before training)")
evaluate_model(model, data_test, args.output_structure, epoch=-1)

# Run training loop.
for epoch in range(args.epochs):
    print(f"Starting epoch {epoch + 1}")

    # Run training epoch.
    print("Training...")
    for batch in data_train.epoch(device):
        loss = compute_loss(model, batch)
        # Perform gradient step.
        loss.backward()
        opt.step()
        opt.zero_grad()
    
    # Compute eval loss and save model.
    print("Evaluating...")
    evaluate_model(model, data_test, args.output_structure, epoch)
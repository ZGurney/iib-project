import argparse
import os
import pickle
import joblib

import lab.torch as B
import matplotlib.pyplot as plt
import numpy as np
import torch
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak

from convcnp import DualConvCNP, ClassConvCNP, RegConvCNP, GPGenerator

# Enable GPU if it is available.
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def split_off_classification(batch):
    """Split off a classification data set."""
    n_context = B.shape(batch["x_context"], 1)
    n_class = np.random.randint(low=1, high=n_context - 1)
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


def compute_loss(model, batch, mode):
    """Compute the sum of the classification and regression loss functions."""
    if mode == "dual":
        class_prob, (reg_mean, reg_std) = model(batch)

        # Clamp the classification probabilities to prevent the loss for NaNing out.
        class_prob = class_prob.clamp(1e-4, 1 - 1e-4)

        class_loss = -B.sum(
            batch["y_target_class"] * B.log(class_prob)
            + (1 - batch["y_target_class"]) * B.log(1 - class_prob)
        )

        reg_loss = 0.5 * B.sum(
            B.log_2_pi
            + B.log(reg_std)
            + ((reg_mean - batch["y_target_reg"]) / reg_std) ** 2
        )

        overall_loss = args.alpha * class_loss + (1 - args.alpha) * reg_loss

    if mode == "classification":
        class_prob = model(batch)

        # Clamp the classification probabilities to prevent the loss for NaNing out.
        class_prob = class_prob.clamp(1e-4, 1 - 1e-4)

        overall_loss = -B.sum(
            batch["y_target_class"] * B.log(class_prob)
            + (1 - batch["y_target_class"]) * B.log(1 - class_prob)
        )

    if mode == "regression":
        (reg_mean, reg_std) = model(batch)

        overall_loss = 0.5 * B.sum(
            B.log_2_pi
            + B.log(reg_std)
            + ((reg_mean - batch["y_target_reg"]) / reg_std) ** 2
        )

    return overall_loss


def take_first(x):
    """Take the first of a batch."""
    if B.rank(x) > 1:
        x = x[0, :, 0]
    return B.to_numpy(x)


# Parse command line arguments.
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--root",
    type=str,
    default="models/model-1",
    help="Directory where models are stored.",
)
parser.add_argument(
    "--file_name",
    type=str,
    default="model1.tar",
    help="Name of the file which stores the checkpoint",
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
args = parser.parse_args()

# Setup working directory.
wd = WorkingDirectory(os.path.join(args.root, "outputs"), seed=0, override=True)

# Setup data generator.
gen_test = GPGenerator(num_tasks=64)

# Construct model.
model = DualConvCNP(small=args.small).to(device)

# Construct optimiser.
opt = torch.optim.Adam(params=model.parameters(), lr=args.rate)

# Load model

#checkpoint = torch.load(os.path.join(args.root, args.file_name), map_location=torch.device("cpu"))
checkpoint = joblib.load(os.path.join(args.root, args.file_name))
epoch = checkpoint["epoch"]
loss = checkpoint["loss"]
model.load_state_dict(checkpoint["state_dict"])
opt.load_state_dict(checkpoint["optimiser"])

model.eval()

# Produce some plots.
print("Plotting...")
batch = gen_test.generate_batch(device)
batch = split_off_classification(batch)
with B.on_device(device):
    # Set `x_target` to a dense linspace for the plots, but save the original
    # target inputs.
    x_target_class = batch["x_target_class"]
    x_target_reg = batch["x_target_reg"]
    batch["x_target_class"] = B.linspace(torch.float32, *gen_test.x_range, 200)
    batch["x_target_reg"] = B.linspace(torch.float32, *gen_test.x_range, 200)
class_prob, (reg_mean, reg_std) = model(batch)

# Plot for classification:
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
plt.savefig(wd.file(f"epoch{epoch + 1}_classification.pdf"))
plt.close()

# Plot for regression:
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
plt.savefig(wd.file(f"epoch{epoch + 1}_regression.pdf"))
plt.close()
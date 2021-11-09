from importlib import reload

import argparse
import os
import joblib

import lab.torch as B
import matplotlib.pyplot as plt
import numpy as np
import torch
import stheno
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak

from convcnp import DualConvCNP, ClassConvCNP, RegConvCNP
import convcnp
convcnp = reload(convcnp)

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

gen_test_comparison = convcnp.GPGenerator2(noise=0, seed=0, batch_size=1, num_tasks=1, num_context_points=50)
test_comparison = gen_test_comparison.generate_batch()

# Sort both the context and target sets
test_comparison["y_context"] = test_comparison["y_context"][:, B.flatten(B.argsort(test_comparison["x_context"], axis=1)), :]
test_comparison["x_context"] = B.sort(test_comparison["x_context"], axis=1)
test_comparison["y_target"] = test_comparison["y_target"][:, B.flatten(B.argsort(test_comparison["x_target"], axis=1, descending=True)), :]
test_comparison["x_target"] = B.sort(test_comparison["x_target"], axis=1, descending=True)

print(test_comparison["x_target"])

test_comparison = split_off_classification(test_comparison, 0.5)

plt.figure()
plt.scatter(
    test_comparison["x_context_reg"],
    test_comparison["y_context_reg"],
    style="train",
    label="Context",
)
plt.scatter(
    test_comparison["x_target_reg"],
    test_comparison["y_target_reg"],
    style="test",
    label="Target",
)
plt.show()

plt.figure()
plt.scatter(
    test_comparison["x_context_class"],
    test_comparison["y_context_class"],
    style="train",
    label="Context",
)
plt.scatter(
    test_comparison["x_target_class"],
    test_comparison["y_target_class"],
    style="test",
    label="Target",
)
plt.show()
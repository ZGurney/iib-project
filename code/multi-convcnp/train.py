import argparse
from time import perf_counter

import lab.torch as B
import matplotlib.pyplot as plt
import numpy as np
import torch
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak

import convcnp
import util

# Enable GPU if it is available.
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

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
util.evaluate_model(model, data_test, epoch=-1)

# Run training loop.
for epoch in range(args.epochs):
    print(f"Starting epoch {epoch + 1}")
    t_start = perf_counter()

    # Run training epoch.
    print("Training...")
    for batch in data_train.epoch(device):
        loss = util.compute_loss(model, batch)
        # Perform gradient step.
        loss.backward()
        opt.step()
        opt.zero_grad()
    
    # Compute eval loss and save model.
    print("Evaluating...")
    util.evaluate_model(model, data_test, epoch)

    t_stop = perf_counter()
    print(f"Epoch {epoch+1} completed in {(t_stop-t_start):,.0f} seconds.")
from posixpath import split
import lab.torch as B
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle

import data_generator

# Enable GPU if it is available.
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def take_first(x, convert_to_numpy=True):
    """Take the first of a batch."""
    if B.rank(x) > 1:
        x = x[0, :, 0]
        if convert_to_numpy:
            x = B.to_numpy(x)
    return x

data_filename = "code/camels_generator/maurer.pickle"
file = open(data_filename, "rb")
data = pickle.load(file)
file.close()

print(data["14400000"].keys())

HydroGenerator = data_generator.HydroGenerator(dict_df=data, 
                channels_c = ['PRCP(mm/day)', 'Q'],
                channels_t = ['PRCP(mm/day)', 'Q'])

batch = HydroGenerator.generate_task()
print(batch.keys())

def split_off_classification(batch, threshold=1.0):
    """Split off a classification data set."""

    def threshold_data(y, threshold):
        y_thresholded = torch.where(y > threshold, 1, 0)
        return y_thresholded

    return {
        "x": batch["x"],
        "y": batch["y"],
        "x_context_class": batch["x_context"],
        "y_context_class": torch.unsqueeze(threshold_data(batch["y_context"][:, :, 0], threshold), 2),
        "x_target_class": batch["x_target"],
        "y_target_class": torch.unsqueeze(threshold_data(batch["y_target"][:, :, 0], threshold), 2),
        "x_context_reg": batch["x_context"],
        "y_context_reg": torch.unsqueeze(batch["y_context"][:, :, 1], 2),
        "x_target_reg": batch["x_target"],
        "y_target_reg": torch.unsqueeze(batch["y_target"][:, :, 1], 2),
    }

batch = split_off_classification(batch)

def plot_data(x_context,y_context,x_target,y_target):
    plt.figure()
    plt.scatter(
        take_first(x_context),
        take_first(y_context),
        label="Context",
    )
    plt.scatter(
        take_first(x_target),
        take_first(y_target),
        label="Target",
    )
    plt.show()

plot_data(batch["x_context_class"], batch["y_context_class"], batch["x_target_class"], batch["y_target_class"])
plot_data(batch["x_context_reg"], batch["y_context_reg"], batch["x_target_reg"], batch["y_target_reg"])

# Create evaluation tasks
gen_test_save = data_generator.HydroGenerator(dict_df=data,
                num_tasks = 64,
                channels_c = ['PRCP(mm/day)', 'Q'],
                channels_t = ['PRCP(mm/day)', 'Q'])
evaluation_tasks = list(gen_test_save.epoch())
print(B.shape(evaluation_tasks))
torch.save(evaluation_tasks, "evaluation-tasks-camels.pt")
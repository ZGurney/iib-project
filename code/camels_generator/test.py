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

def take_first(x, convert_to_numpy=True, output=0):
    """Take the first of a batch."""
    if B.rank(x) > 1:
        x = x[0, :, output]
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

batch = {
    "x": batch["x"],
    "y": batch["y"],
    "x_context_class": batch["x_context"],
    "y_context_class": torch.unsqueeze(batch["y_context"][:, :, 1], 2),
    "x_target_class": batch["x_target"],
    "y_target_class": torch.unsqueeze(batch["y_target"][:, :, 1], 2),
    "x_context_reg": batch["x_context"],
    "y_context_reg": torch.unsqueeze(batch["y_context"][:, :, 0], 2),
    "x_target_reg": batch["x_target"],
    "y_target_reg": torch.unsqueeze(batch["y_target"][:, :, 0], 2),
}

for i in range(2):
    plt.figure()
    plt.scatter(
        take_first(batch["x_context"]),
        take_first(batch["y_context"], output=i),
        label="Context",
    )
    plt.scatter(
        take_first(batch["x_target"]),
        take_first(batch["y_target"], output=i),
        label="Target",
    )
    plt.show()
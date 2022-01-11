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

task = HydroGenerator.generate_task()
print(task.keys())

for i in range(2):
    plt.figure()
    plt.scatter(
        take_first(task["x_context"]),
        take_first(task["y_context"], output=i),
        label="Context",
    )
    plt.scatter(
        take_first(task["x_target"]),
        take_first(task["y_target"], output=i),
        label="Target",
    )
    plt.show()
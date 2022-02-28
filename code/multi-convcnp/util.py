import lab.torch as B
import matplotlib.pyplot as plt
import numpy as np
import torch
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak

def compute_loss(model, batch):
    """Compute the mean of all classification and regression loss functions."""

    def LogSigmoid(x):
        """Compute the log-sigmoid element-wise, without numerical issues"""
        c = B.max(-x, 0)
        return -c - B.log(B.exp(-c) + B.exp(-x - c))

    predictions = model(batch)

    losses = []

    for index, output in enumerate(batch):
        z = predictions[index]

        if output["type"] == "classification":
            losses.append(-B.sum(
                output["y_target"] * LogSigmoid(z)
                + (1 - output["y_target"]) * LogSigmoid(-z)
            ))
        else:
            reg_mean = z[:, :, :1]
            reg_std = B.exp(z[:, :, 1:])

            losses.append(0.5 * B.sum(
                B.log_2_pi
                + B.log(reg_std)
                + ((reg_mean - output["y_target"]) / reg_std) ** 2
            ))

    assert len(losses) == len(batch), f"Size of loss vector is not {len(batch)}"

    losses = B.to_numpy(losses)
    overall_loss = np.mean(losses)

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
        """
        # Set up output to compute loss on single task
        single_task = {}
        for key, value in output.items():
            if key != "type":
                single_task[key] = take_first(value, convert_to_numpy=False)

        # NOTE: add square brackets to make it look like a batch with multiple outputs?
        loss = compute_loss(model, single_task)
        print(f"Output {output_index} loss: {loss:6.2f}")
        """

        with B.on_device(device):
            # Set `x_target` to a dense linspace for the plots, but save the original
            # target inputs.
            x_target = output["x_target"]
            output["x_target"] = B.linspace(torch.float32, *gen_test.x_range, 200)

            # NOTE: indexing at output index to extract single prediction
            prediction = model([output])[0]

        plt.figure()
        plt.title(f"Output {output_index} (Epoch {epoch+1})")
        plt.scatter(
            take_first(output["x_context"]),
            take_first(output["y_context"]),
            style="train",
            label="Context",
        )
        plt.scatter(
            take_first(x_target),
            take_first(output["y_target"]),
            style="test",
            label="Target",
        )
        if output["type"] == "classification":
            prediction = B.sigmoid(prediction)
            plt.plot(
                take_first(output["x_target"]),
                take_first(prediction),
                style="pred",
                label="Prediction",
            )
        else:
            (reg_mean, reg_std) = (prediction[:, :, :1], B.exp(prediction[:, :, 1:]))
            plt.plot(
                take_first(output["x_target"]),
                take_first(reg_mean),
                style="pred",
                label="Prediction",
            )
            plt.fill_between(
                take_first(output["x_target"]),
                take_first(reg_mean - 1.96 * reg_std),
                take_first(reg_mean + 1.96 * reg_std),
                style="pred",
            )
        tweak(legend_loc="best")
        plt.savefig(wd.file(f"epoch{epoch+1}_{i}_output{output_index}.pdf"))
        plt.close()

# Evaluation script
def evaluate_model(model, evaluation_batches, epoch):
    # Compute overall loss
    losses = [compute_loss(model, batch) for batch in evaluation_batches]
    losses = B.to_numpy(losses)
    error = 1.96 * np.std(losses) / np.sqrt(len(losses))
    print(f"Overall loss: {np.mean(losses):6.2f} +- {error:6.2f}")

    # Plot outputs for each batch in the evaluation tasks
    print(f"Plotting {len(evaluation_batches[0])} tasks")
    for i, batch in enumerate(evaluation_batches):
    	plot_graphs(batch.copy(), epoch, i)

    # Save checkpoint
    checkpoint = {
        "epoch": epoch + 1,
        "loss": (np.mean(losses), error),
        "state_dict": model.state_dict(),
        "optimiser": opt.state_dict(),
    }

    torch.save(checkpoint, f"./outputs/model{epoch+1}.tar")
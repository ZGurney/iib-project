import lab.torch as B
import matplotlib.pyplot as plt
import numpy as np
import torch
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak

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
        if output["type"] == "classification":
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
def evaluate_model(model, epoch, evaluation_batches):
    # Compute overall loss
    losses = [compute_loss(model, batch) for batch in evaluation_batches]
    losses = B.to_numpy(losses)
    error = 1.96 * np.std(losses) / np.sqrt(len(losses))
    print(f"Overall loss: {np.mean(losses):6.2f} +- {error:6.2f}")

    # Plot outputs for each batch
    print(f"Plotting {len(evaluation_batches[0])} outputs")
    for i, batch in enumerate(evaluation_batches):
    	plot_graphs(batch, epoch, i)

    # Save checkpoint
    checkpoint = {
        "epoch": epoch + 1,
        "loss": (np.mean(losses), error),
        "state_dict": model.state_dict(),
        "optimiser": opt.state_dict(),
    }

    # Save the trained model in the outputs folder
    #model_file_name = f"model{epoch + 1}.tar"
    #with open(model_file_name, "wb") as file:
    #    joblib.dump(value=checkpoint, filename=os.path.join('./outputs/', model_file_name))

    torch.save(checkpoint, f"./outputs/model{epoch+1}.tar")
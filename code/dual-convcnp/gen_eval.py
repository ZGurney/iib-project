# Generate evaluation tasks

import lab.torch as B
import torch
import stheno

import convcnp

def generate_eval_tasks(kernel, num_batches):
    """
    Generates a single task with specified kernel and seed
    """
    num_tasks = 16 # number of tasks per batch
    gen_test_save = convcnp.GPGenerator2(kernel=kernel, noise=0, seed=1, num_tasks=num_batches*num_tasks)

    evaluation_tasks = list(gen_test_save.epoch())
    
    # Sort both the context and target sets

    def sort_y(x,y, descending):
        y_sorted = B.zeros(y)
        for i in range(num_tasks):
            num_points = B.shape(y)[1]
            indices = B.flatten(B.argsort(x, axis=1, descending=descending))[i*num_points:(i+1)*num_points]
            y_sorted[i, :, :] = torch.unsqueeze(y[i, :, 0][indices], 1)
        assert B.shape(y) == B.shape(y_sorted), f"shapes do not match: {B.shape(y)} is not equal to {B.shape(y_sorted)}"

        return y_sorted
    
    for batch in evaluation_tasks:
        batch["y_context"] = sort_y(batch["x_context"], batch["y_context"], False)
        batch["x_context"] = B.sort(batch["x_context"], axis=1)
        batch["y_target"] = sort_y(batch["x_target"], batch["y_target"], True)
        batch["x_target"] = B.sort(batch["x_target"], axis=1, descending=True)

    return evaluation_tasks
        
evaluation_tasks = generate_eval_tasks(stheno.EQ().stretch(0.25), 4)
torch.save(evaluation_tasks, "evaluation-tasks.pt")
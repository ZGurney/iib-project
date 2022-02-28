# Daily Note

16 February 2022

## Current areas of work

1. Benchmark on the CAMELS dataset
2. Extending to handle multiple outputs
3. Examining time delays

## Ideas for handling the generated CAMELS meta-dataset

- Problem: DualConvCNP is finding it difficult to model bias offsets in the streamflow data, which can range from around 10 to 500
- Inside `convcnp.py`, add an algorithm that subtracts the mean from each task and then adds it back on at the end

### Algorithm suggestion in `convcnp.py`

- The forward function takes `batch` as input which has the following structure: `{"y_class_context": …, "x_class_context": …, …}`, of each each key is assigned a data structure `[num_tasks_per_batch, num_datapoints_per_task, 1]`
- For the two keys, `"y_reg_target"` and `"y_reg_context"`, access the corresponding data
	- Initialise an empty matrix using something like `"B.zeros(batch["y_reg_context"])"`
	- Compute the task-specific mean across the "rows" or over each task to yield a 16x1 vector
	- Repeat this vector `num_datapoints_per_task` times to form a *means matrix*
	- For the two matrices associated with the regression context and target outputs, subtract the *means matrix* to form a zero-meaned regression task-specific batch
- Before returning `z_reg`, add the *means matrix* to create the required offset

## Summary of work done on the multi-output ConvCNP

- Used a data structure to handle the multiple outputs as follows:
	- Each batch has `num_outputs` components, whether classification or regression
	- For each component, they have a dictionary structure `{"type", "x_context": …, "y_context: …, "x_target": …, "y_target: …}`
	- For each item, it follows the same structure of `[num_tasks_per_batch, num_datapoints_per_task, 1]`
	- However, there is now an additional item in the dictionary which tells us the type of data in the output, either `"classification"` or `"regression"`
- Key modifications:
	- `convcnp.py`: heavily changed the forward function to handle any number of outputs
		- problems with the `convert_batched_data()` function was solved by adding an if statement to check if operating on a string, i.e. the `"type"` item
		- still experiencing problems with the concatenation operation before the CNN
		- now returns a 4-dimensional matrix called `z_outputs`: first dimension is the output, then the rest are as usual
	- `data_shift.py`: now returns batch with two components, corresponding to the classification output and the regression output, in that order
	- `encoder.py` and `decoder.py` remain the same since they operate on each output independently
	- Changed the `compute_loss()`, `plot_graphs()` and `evaluate_model()` to handle multiple outputs
		- One central problem is Python's default not to copy variables which was largely solved by using `batch.copy()`, however, still not sure how effective it was

## Summary of work on time delayed dual outputs

- Two areas of interest:
	1. How large of a time delay can we model?
	2. Can the DualConvCNP model randomly sampled time shifts, in other words, if the time-shift is not constant between tasks and alternates randomly between `0.5` and `1.0`, when the context sets are given, can it infer the latent time-shift parameter and then correspondingly make predictions?
1. Advances in area 1
	- Seems that a smaller or larger CNN structure (meaning both greater depth and breadth in the network structure) does not affect performance
	- Seems that `1.5` is the maximum time delay before predictions become meaningless. Is this a general result?
2. Advances in area 2
	- From initial results, seems like this could be possible, however, we made the task exceedingly challenging for the DualConvCNP as we limited the overlap domain between the two context sets to less than `1.0`
	- [ ] Try using a set of evaluation tasks with context data across the entire range of the regression output, but only at the start of the context output and examine performance on the second half of the x-domain as the target region
	- Broader question of how to rigorously perform an evaluation of the DualConvCNP on different time shifts, since greater time shifts imply that the overlapping region where data are correlated becomes smaller and thus worse performance

## Some future work

- Doing some research and asking Wessel, perhaps arrange a call:
	- causal filters and how to implement with GPs
	- mechanical systems for further performance benchmarking
	- Bernouilli-Gamma mixtures for the precipitation
- Asking Rich to go through Gaussian Neural Processes and how we can implement
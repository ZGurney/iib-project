#meeting
# Daily Note

21 October 2021

## Meeting with Wessel

Where should I run my models?

- Azure compute seems to be fine, he has used AWS.

What is a "batch"?

- Split up the thousands of tasks into sets of tasks to be trained together.
- The size of the batch is set in `convcnp.py`.

Do we sub-sample the same number of context and target points?

- Can have, say, 15 context points and 10 target points but these must be consistent across the batch so that the tensors of each task can be properly concatenated into a batch.

Should I combine or keep separately the dual ConvCNP and the ordinary ConvCNP?

- The ordinary ConvCNP can be trained within the dual code, just requires some modifications in the `convcnp.py` file.

How does the dual ConvCNP work with multi-output data?

- We produce a density and data channel for each output.
- We then stack these vectors (overall four) and pass them through a CNN.
- The CNN produces three stacked vectors which encode:
	1. Classification probabilities
	2. Regression means
	3. Regression variances
- Thus, we then separate out the vectors, the first for the classification and the last two for the regression, and decode separately.

Can we generate uncertainty in the classification case?

- The uncertainty is simply the class probability which is sufficient.
- Could additionally take a Bayesian view but not necessary.

What is the structure of the data?

- Three dimensions: tasks in a batch, channels, datapoints.

How can I save my models?

- Since we used standard PyTorch modules, can use their approach.
- [ ] Read [PyTorch tutorial](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

How to solve the NaN problem?

- Need to clamp numbers which overload the Float32 format by setting an upper and lower bound, determined experimentally.
- Two possible problems:
	1. `reg_std` in `train.py:52`
	2. `B.exp(…)` in `convcnp.py: 84`
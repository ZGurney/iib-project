# Daily Note

7 January 2022

## Removing datapoints on sides of two outputs
- Aim: induce artificial delay $\tau$  in continuous output to mimic real world data where high rainfall in days before high water level is likely relation
- Task: Remove datapoints at end of continuous output and start of binary output
	- Continuous: remove end from `x_range(1)-tau` onwards
	- Binary: remove start until `x_range(0)+tau`
- First method
	- Generate sampled function as usual from GP
	- Generate continuous and binary output with context and target x-y pairs
	- Create function `remove_datapoints(x_values, y_values, location)`
		- Identify x-values which are larger or smaller than specified value
		- Remove those x-values from the array
		- Identify y-values which correspond to those x-values
		- Remove those y-values
	- Run function over four arrays: continuous context, continuous target, binary context, binary target
	- *Possible problem: will generate individual tasks with different sizes*
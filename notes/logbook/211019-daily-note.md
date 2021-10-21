# Daily Note

19 October 2021

- Wessel has now resolved the device bug by updating one of the dependencies; the code now runs perfectly locally and on Azure
- Should I use native Azure or MLFlow for logging metrics?
	- Seems that MLflow is a more general form for logging if someone else wanted to run my code

## Questions for Wessel

- Running on Azure
	- Run on Azure or CBL compute?
	- How to make sure I am not incurring additional costs?
	- Current access to "Standard_NC6 (6 cores, 56 GB RAM, 380 GB disk), GPU - 1 x NVIDIA Tesla K80" at **$1.28/hr**
	- Problem with NaNing out at epoch 20
- Dual ConvCNP
	- What is in a batch?
	- Do we sub-sample the same number of context and target points?
	- What does `take_first(x)` do with the rank thing?
	- Should I combine or keep the dual ConvCNP and ordinary ConvCNP code?
	- Generating uncertainty for classification?

- Parameters
	- Exponentiated quadratic (EQ) kernel, length-scale 0.25
		- use weakly periodic and matern -5/2 parameters from ConvCNP paper?
	- No. of context and target points for batch uniformly sampled between 3 and 50
	- Input locations uniformly from -2 to 2
	- Default: Adam optimiser, weight decay $10^{-5}$, small CNN architecture
	- **Do they match the ConvCNP original parameters?**

Plan
- performance basic:
	- losses together or separate?
- missing data in classification and regression
- different kernels
	- How to generate samples from different kernels?
- related kernels
- more than one output
	- Where are the representations stacked?
- different architectures for handling multi-output


Plotting
- True function: sample from the GP prior, which originally generated the task datapoints
- (Ground truth GP: GP posterior distribution using exact kernel and then performing inference conditioned on the context set)
- Neural process predictive distribution
- Context and target sets


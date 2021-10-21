# Daily Note

18 October 2021

- Cannot create GPU compute on Azure ML
- Still encountering same problem as mentioned in [[211014-daily-note]] when running `train.py`

## Meeting with Rich

- Wessel
	- write email specifying the computational bug
	- help with plotting loss curves and necessary plots

- Initial configuration
	- ConvCNP
	- two outputs
	- sampled from same underlying function
	- generated from EQ kernel

- Steps after initial implementation
	- change kernel, just like in normal research papers
	- remove datapoints at certain input locations in one dataset to see if multi-output version can model correlation between datapoints in the other dataset, for example temperature and snow
	- How many outputs are we expecting to use?
		- In medical situations, might have very high-dimensional data like an image of a body; this is out of scope of our project
		- We will limit ourselves to the order of $10^1$ to be handled
	- work with real data: speak with Ana and Mark mid-term (W4) who work on climate prediction on structure of datasets and missing points

- ConvCNP vs GNP
	- [[gaussian-np|Gaussian neural processes]] being introduced in [pre-publication paper](https://openreview.net/forum?id=3pugbNqOh5m)
	- uses same ConvDeepSet structure but now parameterises the covariance function which can capture correlations across input space
	- **Key point**
		- Currently working with independent noise models across outputs, i.e. the noise associated with one output prediction is independent of another output prediction
		- However, we might expect dependencies across outputs and could use a correlated noise model
			- for example, if we under predict the temperature in a given location, it is likely to be correlated with an over prediction of snow in that same location
		- Thus, this modelling of dependencies across outputs is closely linked to modelling dependencies across input space, which Gaussian neural processes solve
		- _Are we distinguishing between noise uncertainty and functional uncertainty?_

- Structure of the masters thesis
1. Application
2. Technique
3. Limitation
- successes and problems in the synthetic case should be reflected in the real case

### Next steps
- [ ] Plot synthetic data from the GP
- [ ] Make predictions without training
- [ ] Show how training improves performance
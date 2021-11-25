# Michaelmas presentation

14.00, 24 November 2021

## Brief

- recommend 6 slides. 1.5 mins each
- Include
	- Aims of the project and their significance
	- Background and context
	- Plan of how to achieve the aims
	- Progress made so far towards fulfilling the plans

## Notes

Structure
- Why are they important?
- How do they work?
- What progress have we made thus far? 

Motivation
- If we have observations over time at different locations for:
- Observed flow of river
- Rainfall
- If sensor 1 fails, can sensor 2’s observations inform predictions over the missing outputs for sensor 1?

-   Irregularly sampled time-series in healthcare settings
-   Off-the-grid spatial data in environmental settings
-   Current approaches cannot handle multi-output data
-   Examples

-   Healthcare: using different measurements like heart rate, blood pressure, temperature
-   Spatial-data weather prediction problem: using different measurements like rainfall, temperature, wind speeds, precipitation

-   Only a subset of observations available at each input locations and we wish to make joint predictions over all of them

Comparison with traditional approach of supervised learning
- Define a task $\mathcal{D} := (\mathcal{D}_c, \mathcal{D}_t)$
- Neural network
	- Train on single dataset $\mathcal{\D}=$ that maps a dataset to a function $f_\theta(x)$
	- Test on target inputs
- Learning common structure across related but different tasks


## Possible questions

- Where can this be applied? What kinds of datasets will you explore?
- Why not use *x* approach to handle multi-output data?
- What kinds of correlations can the dual ConvCNP model?
- What are the limitations of our model?
- What happens in the encoder and decoder step?
	- What is the difference between ordinary deep sets and convolutional deep sets? Embeds into a function which is in an infinite-dimensional space and thus translation equivariant rather than a fixed-dimensional vector
- What other architectures exist to handle multi-output data? Is there an optimal one?
- What type of neural process did you use? What kind of encoder and decoder did you choose?
	- ConvCNP: translation equivariance, computationally lighter than LNP
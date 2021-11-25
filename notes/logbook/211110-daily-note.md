# Daily Note

10 November 2021

## Meeting with Marc

- Explored possible datasets to use multi-output ConvCNP
	- Forest fires -> Anna is currently using multi variate correlation ConvCNP
	- River flow
	- Air quality
- River flow dataset
	- Precipitation, river flow, temperature, solar radiation, basin type time series at different locations
	- Make predictions
		- regions of missing data, say where a sensor has been taken offline
		- into the future
	- Complex patterns of missing data requires composing "primitives" to pre-process data
	- Likely benchmark against a GP which is just about appropriate for this situation
	- [ ] Meet with Marc to obtain dataset
- Air quality
	- PM2.5, PM10.0, NOx levels time series at different locations
	- Similar to river flow dataset with predictions
	- [ ] Rich to message Michelle and Alex about possible datasets

## Meeting with Rich

- Future: context set induces dependencies but could introduce correlated errors like in convGNP
	- Variational inference layer at end to deal with latent variable


- [ ] Solve loss metrics
- [ ] Plot ground truth -> necessary?
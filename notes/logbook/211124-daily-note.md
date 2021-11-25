# Daily Note

24 November 2021

## Presentation questions

- Under what scenarios would the multi-output neural process fail?
	- Consider sparsity in number of related tasks
	- Complex relation between outputs?
	- Should show good performance as can handle different patterns of "data-missingness" and uses neural networks to parameterise complex interrelations
- Why is loss not significantly different?
	- [ ] Need to generate evaluation tasks with more extreme cases of missingness


## Meeting with Marc on CAMELS dataset

- Broad characteristics
	- 671 locations
	- Time series gathered over 30 years from 1980 to 2010
	- Meteorological time series, sometimes called forcings
	- Real and simulated baseline predictions of streamflow
	- Catchment attributes for each location
- Need a data generator to create tasks by:
	- 1. Choose random location
	- 2. Choose random period of time with a fixed length (30 days, 1 year)
	- 3. Sample some context points, sample some target points

## Work today

- Working on creating general model that handles any number of classification and regression outputs to prepare for real world data

## Some tasks

- [ ] Try random meta-sampling between negative and positive correlations
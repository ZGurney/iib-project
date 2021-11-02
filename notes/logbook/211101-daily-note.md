# Daily Note

1 November 2021

## Tasks

- [ ] Write email to Wessel
	- How to save model? Why are model parameters different?
	- How to solve log-sigmoid problem?
	- How to seed GP generated data?

## Notes

- Solved log-sigmoid numerical problem using log-sum-exp trick
	- See solution [here](https://github.com/pytorch/pytorch/pull/2211)
	- $\log(\frac{1}{1+e^{-x}}) = $
- Use same 64 tasks for evaluation
	- Print error at each iteration
	- When evaluating average loss, set 4 batches with 0.2, 0.4, 0.6, 0.8: `batches = [split_classification(batch, proportion_class=a) for a in range(0.2, 0.8, step=0.2)]` then later on can plot performance on `take_first(batches)`
	- Plot oracle GP and original function
	
	1012
	
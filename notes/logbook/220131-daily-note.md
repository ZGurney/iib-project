# Daily Note

31 January 2022

## Solving the evaluation task generator

- Identified problem is generating incorrect shapes of the `y_context` and `y_target` tensors
- Solved problem by indexing over smaller array, then using `torch.unsqueeze()` to add in the extra dimensions
- Run all three experiments, note may have to do with noise!
- Modified `data_shift.py` to include disjoint context and target set generation for evaluation of performance

## Solving the time shift generator

- `plot_graphs()` function was modifying the batches in `evaluation_tasks` directly when we create a grid of 200 points
- We cannot modify the dense x grid generation in plotting graphs as we need to modify the batch itself to generate predictions over that dense grid

## CAMELS data generator

- Need to later implement seeded version using NumPy random number generator
- Need to solve x_range problem
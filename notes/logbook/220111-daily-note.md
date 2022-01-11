# Daily Note

11 January 2021

## Preparing HydroGenerator

- Removed `task['y_target'] = task['y_target'][:,:,0].unsqueeze(dim=2)` from `task_preprocessing.py` script that caused problem with missing outputs in `y_target`
- Found error in taking sample longer than timeslice of 60 days `Cannot take a larger sample than population when 'replace=False'` when defining `x_ind`
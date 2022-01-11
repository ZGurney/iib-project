# Daily Note

10 January 2022

## HydroGenerator

- Using Marc's `maurer.pickle` data
	- Format is dictionary with keys like `14400000` corresponding to each river basin
	- Then each river basin is a dictionary with the following keys `['Year', 'Mnth', 'Day', 'Hr', 'Dayl(s)', 'PRCP(mm/day)', 'SRAD(W/m2)',
       'SWE(mm)', 'Tmax(C)', 'Tmin(C)', 'Vp(Pa)', 'basin', 'Q', 'qc']`
- Modifications
	- Removed `y_target_val`
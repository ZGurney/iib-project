# Stheno

Stheno is a Python package for using Gaussian processes for modelling data.

## Some basic documentation

```python
pip install stheno # Installation

# Note that `x` is the input space and `x_obs` is where we sample synthetic datapoints

f = GP(mean=0, kernel, measure=prior) # Create a Gaussian process prior

f(x) # Calls the Gaussian process f to construct a finite-dimensional distribution at inputs x
f(x, noise) # Same as above, but now includes variance `noise`

mean, lower, upper = f(x).marginal_credible_bounds() # Compute mean function plus marginal lower and upper 95% central credible region bound functions at inputs x

f(x).sample(n=1) # Draw samples from the Gaussian process

f_post = f | (f(x_obs, noise), y_obs) # Compute posterior by conditioning the prior on the observations
```

## Questions for Wessel

What is the difference between:
```python
f(x).sample()
f.measure.sample(f(x))
```

## References

- [Official manual](https://github.com/wesselb/stheno) by Wessel
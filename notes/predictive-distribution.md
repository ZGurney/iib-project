---
aliases: predictive distribution
---
# Predictive distribution

The predictive distribution $p(\mathbf{y}_t | \mathbf{x}_t; D_c)$ is a stochastic process over possible predictors. We can parameterise this in two common ways:

1. Univariate Gaussian distributions at each target input as in the [[cnp-family|conditional neural process family]]: $p(\mathbf{y}_t | \mathbf{x}_t; D_c) = \prod_{t=1}^T \mathcal{N}(y^{(t)} | \mu_t(D_c), \sigma_t(D_c))$
2. Multivariate Gaussian distribution over all target inputs as in [[gaussian-np|Gaussian neural processes]]:  $p(\mathbf{y}_t | \mathbf{x}_t; D_c) = \mathcal{N}(\mathbf{y}_t | \mathbf{m}, \mathbf{K})$ where $m$ is a mean vector and $K$ is the covariance matrix
	- This addresses the problem of incoherent samples in [[cnp-family|conditional neural processes]] but is more computationally demanding

## References

- @garnelo2018
- @anonymous2021
---
aliases: conditional neural processes, conditional neural process family
---

# Conditional neural process family

A subfamily of [[neural-processes]] which employ a factorisation assumption for modelling the predictive distribution: $p_\theta(\mathbf{y}_t | \mathbf{x}; D_c) = \prod_{t=1}^T p_\theta(y^{(t)} | x^{(t)}; D_c)$.

Although easy to train by maximum-likelihood, their main problem is that they are unable to model correlations across input space and thus generate incoherent samples.

## References
- @garnelo2018
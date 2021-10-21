# SetConv operation

Although convolutional neural networks (CNNs) have the translation equivariance property, they take discrete signals as their inputs whereas we wish to work with context sets that have continuous input-output pairs $D_c = \{(x^{(c)}, y^{(c)})\}_{c=1}^C$. Thus, Gordon et al. introduce a SetConv layer that takes the set of pairs as input and outputs a function with continuous inputs $x$, defined as follows:
$$\text{SetConv}(D_c)(x) = \sum_{c=1}^C \begin{bmatrix} 1 \\ y^{(c)} \end{bmatrix} \omega_\theta(x-x^{(c)})$$

Note: $\omega_\theta$ is a function that is usually set to be an exponentiated-quadratic (EQ) kernel $\omega_\theta(\cdot) = \exp(-\frac{\|\cdot\|_2^2}{l^2})$ where $l$ is a length-scale parameter. Also, as a result of the $\begin{bmatrix} 1 \\ y^{(c)} \end{bmatrix}$ term, we have an additional _density channel_ to distinguish between the cases with no observed datapoint and $y=0$ at an input location.

Thus, we preserve the permutation invariance property essential for treating the dataset appropriately.

## References

@gordon2019
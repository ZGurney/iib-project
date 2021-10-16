# Convolutional conditional neural processes (ConvCNP)

A member of the [[cnp-family]] that addresses the issue of failing to generalise outside the training regime by introducing the inductive bias of translation equivariance (TE). Translation equivariance means that if the context dataset is shifted in input space by an amount $T_\tau$, then the resulting predictions should also be shifted by $T_\tau$.

## Architecture of the ConvCNP

### SetConv layer

Although convolutional neural networks (CNNs) have the translation equivariance property, they take discrete signals as their inputs whereas we wish to work with context sets that have continuous input-output pairs $D_c = \{(x^{(c)}, y^{(c)})\}_{c=1}^C$. Thus, Gordon et al. introduce a SetConv layer that takes the set of pairs as input and outputs a function with continuous inputs $x$, defined as follows:
$$\text{SetConv}(D_c)(x) = \sum_{c=1}^C \begin{bmatrix} 1 \\ y^{(c)} \end{bmatrix} \omega_\theta(x-x^{(c)})$$

Note: $\omega_\theta$ is a function that is usually set to be a radial basis function (RBF) $\omega_\theta(\cdot) = \exp(-\frac{\|\cdot\|_2^2}{l^2})$ where $l$ is a length-scale parameter. Also, as a result of the $\begin{bmatrix} 1 \\ y^{(c)} \end{bmatrix}$ term, we have an additional _density channel_ to distinguish between the cases with no observed datapoint and $y=0$ at an input location.

Thus, we preserve the permutation invariance property essential for treating the dataset appropriately.

### Encoder-decoder architecture

After the SetConv operation on the context set we discretise the output continuous function such that it can be passed to a CNN in the next step. We then obtain the continuous representation $R(\cdot)$ by passing the CNN's output through another SetConv operation.

The decoder obtains a target-specific representation $R(x^{(t)})$ to then be passed through a multi-layer perceptron (MLP) to obtain the final predictive distribution.

![Computational graph of ConvCNP by Dubois et. al](https://yanndubs.github.io/Neural-Process-Family/_images/computational_graph_ConvCNPs1.png)
_Computational graph of ConvCNP by Dubois et. al_

## References

- @dubois2020
- @gordon2019
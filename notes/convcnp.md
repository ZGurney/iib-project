# Convolutional conditional neural processes (ConvCNP)

A member of the [[cnp-family|conditional neural process family]] that addresses the issue of failing to generalise outside the training regime by introducing the inductive bias of [[translation-equivariance|translation equivariance]] (TE).

## Architecture of the ConvCNP

After the [[SetConv]] operation on the context set we discretise the output continuous function such that it can be passed to a CNN in the next step. We then obtain the continuous representation $R(\cdot)$ by passing the CNN's output through another SetConv operation.

The decoder obtains a target-specific representation $R(x^{(t)})$ to then be passed through a multi-layer perceptron (MLP) to obtain the final predictive distribution.

![Computational graph of ConvCNP by Dubois et. al](https://yanndubs.github.io/Neural-Process-Family/_images/computational_graph_ConvCNPs1.png)
_Computational graph of ConvCNP by Dubois et. al_

## References

- @dubois2020
- @gordon2019
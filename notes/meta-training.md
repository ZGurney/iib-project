# Meta-training for neural processes

For [[neural-processes]], we require a set of datasets or "tasks" $\{\mathcal{D}_i\}_{i=1}^{N_\text{tasks}}$, known as a meta-dataset. Since neural processes learn how to produce a predictive distribution from a context set, the tasks consist of independent samples of functions from the underlying data-generating processes.

1. Sample a task $\mathcal{D}$ from the meta-dataset
2. Randomly split the tasks into a context set  $\mathcal{D}_c$ and a target set $\mathcal{D}_t$ where $\mathcal{D} = \mathcal{D}_c \cup \mathcal{D}_t$
3. Pass the context set through the neural process to obtain the predictive distribution $p_\theta(\mathbf{y}_t | \mathbf{x}_t;\mathcal{D}_c)$ at the target inputs $\mathbf{x}_t$
4. Measure performance of predictions over the target set by computing the log-likelihood $\mathcal{L} = \log p_\theta(\mathbf{y}_t | \mathbf{x}_t;\mathcal{D}_c)$
5. Compute the gradient $\nabla_\theta\mathcal{L}$


## References

- @dubois2020
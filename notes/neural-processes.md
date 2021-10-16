# Neural processes

Neural processes tackle issues of working in a small-data regime with good uncertainty estimation by meta-learning a mapping from datasets to a predictive distribution.

The neural process family draws inspiration from ideas of deep learning which is excellent at approximating complex function as well as from probabilistic approaches which use prior knowledge to perform inference in small-data regimes. To contrast the two perspectives, deep neural networks are trained from scratch on a large single dataset to learn a function that can be used as a predictor on unseen data; neural processes are trained on a set of related datasets and incorporate uncertainty through learning a distribution over predictions.

## References
- @garnelo2018
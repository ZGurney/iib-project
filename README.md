# IIB Project

- Course: [MEng in Engineering](http://teaching.eng.cam.ac.uk/node/339) at the University of Cambridge
- Subject: Multi-output neural processes
- Supervisor: [Professor Richard Turner](http://www.eng.cam.ac.uk/profiles/ret26)

## Aims of the project

1. Build a conditional neural process (CNP) for handling multi-output data and deploy on both synthetic and real-world data.
2. Understand which architectures are good for deploying CNPs on multi-output data, both on the input and output side.
3. Establish benchmarks for testing and evaluating CNPs for multi-output data, including health and environmental science datasets.

## What is a neural process?

Deep learning methods have achieved state-of-the-art performance across a range of tasks including speech recognition, computer vision and recommendation systems. However, they fall short in certain problems where only small datasets are available and good uncertainty estimation is required, for example in medical diagnosis scenarios. Neural processes tackle both issues, identified above, by meta-learning a mapping from datasets to a predictive distribution.

The neural process family draws inspiration from ideas of deep learning which are excellent at complex function approximations as well as probablistic approaches which use prior knowledge to perform inference in small-data regimes. To contrast the standard supervised learning approach with neural processes: the former are trained from scratch on a single dataset to learn a function that can be used as a predictor on unseen data; the latter are trained on a set of related datasets and incorporate uncertainty through learning a distribution over predictions.


## Key references

1. [Original paper on conditional neural processes](https://proceedings.mlr.press/v80/garnelo18a.html) by Garnelo et. al in 2018
2. [Online book on the neural process family](https://yanndubs.github.io/Neural-Process-Family) by Yann Dubois, Jonathon Gordon and Andrew Foong in 2020

# Daily Note

11 October 2021

## Meeting with Rich

_Met Rich at 10.00 to 11.00 on Monday 11 October 2021 to review fundamental theory underpinning project, including:_

### 1. What kind of neural process will we be using in this project?

  - [[cnp-family|Conditional neural processes]] (CNP) are easy to work with out-of-the-box, however, due to their factorisation assumption, they cannot generate consistent samples from the predictive distribution -- a problem in many datasets where nearby data points are correlated, for example, temperature over a certain spatial region.
  - Latent neural processes, sometimes referred to in literature simply as neural processes, solve the problem of incoherent samples by treating the representation as a latent variable. However, these can be computationally demanding to evaluate and do not scale well.
  - Therefore, one mid-way solution is to use [[gaussian-np|Gaussian neural processes]]  which define the predictive distribution as a multivariate Gaussian, parameterising the mean vector and the covariance matrix whose off-diagonal terms allow correlation between data points to be captured.
  - In addition, we will be using a [[convcnp|convolutional conditional neural process]] (ConvCNP) as experience has shown that, although more complex in structure, tend to be easier to train.

### 2. How do we generate the synthetic data to benchmark the handling of multi-output data?

  - As a simplest first step, we generate both a regression and classification dataset as follows. An underlying ground-truth function is sampled from a Gaussian process for the regression dataset and we apply a threshold at a certain value, above and below which we assign two classes.
  - Later on, instead of using an identical underlying function, two different but related ground-truth functions could be generated to mimic more closely real life situations.
  - Of course, once we show that the multi-output model outperforms models that can handle only single output data on these synthetic datasets, we can evaluate its performance on real datasets.

### 3. What kind of encoder-decoder architecture will we use to handle the multi-output data?

  - Deep sets result requires representation of the context set to take a certain form to ensure the encoder is invariant under permutations of the context set.
- Thus, we have multiple possible architectures for combining the different outputs. For now, we will start with the architecture that computes representation vectors for each output, then stacks them into an overall vector and perform the decoding step as usual.

### Next steps

#### This week

- [ ] Organise a meeting with Wessel to understand his [approach for handling multi-output data](https://github.com/wesselb/gabriel-convcnp).
- [ ] Set up a basic working version of the multi-output ConvCNP and generate some simple plots.
- [ ] Set up an ordinary ConvCNP to train on a single dataset for the regression and classification case.
- [ ] Assess performance of the multi-output ConvCNP against the ordinary ConvCNP baseline.

#### Later

- [ ] Use different but related underlying functions to generate the different outputs and see if the multi-ouput ConvCNP captures the correlation.
- [ ] Increase the number of outputs beyond two and try different permutations of regression and classification datasets.
- [ ] Consider different possible architectures for handling the representation of the multi-output data.
- [ ] Test the performance on real datasets, including possibly some from climate science and health.

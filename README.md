# Neural networks

This repository is an exploration into deep neural networks, starting from first principles.

## Supervised learning
We aim to build a model that takes an input $\mathbf x$ and produces a prediction $\mathbf y$. The model also contains parameters $\phi$, so:

$$
\tag{1}
\mathbf y = f[\mathbf x, \phi].
$$

When we talk about *learning* or *training* a model we talk about finding parameters $\phi$ that make sensible output predictions from the input. We learn this from using a *training dataset* of $I$ pairs of input and output examples $(\mathbf x_i, \mathbf y_i)$.

To determine how well a model is performing we define a scalar value, called *loss* $L$, that summarizes how poor a model is performing. We define the loss as a function $L[\phi]$ of the parameters. When we train the model we are seeking parameters $\hat \phi$ that minimizes the loss function:

$$
\tag{2}
\hat \phi = \underset{\phi}{\text{argmin}} \left[ L[\phi] \right].
$$

After training a model, we assess its performance using a separate *test dataset* to see how well it generalizes to examples that it didn't observe during training.

## Shallow neural networks
### Universal approximation theorem
In this section we will generalize the example with three hidden units. Let $D$ be the number of hidden units where the $d^\text{th}$ hidden unit is:

$$
h_d = a[\theta_{d0} + \theta_{d1}x],
$$

and these are combined linearly to create the output:

$$
y = \phi_0 + \sum\limits_{d=0}^D \phi_d h_d.
$$

The number of hidden units in a shallow network is a measure of the *network capacity*. With ReLU activation functions the output of the network with $D$ hidden units has at most $D$ joints and so is a piecewise linear function with at most $D+1$ linear regions.

With enough capacity (hidden units), a shallow network can describe any continuous 1D function defined on a compact subset of the real line to arbitrary precision. The *universal approximation theorem* proves that for any continuous function, there exists a shallow network that can approximate this function to any specified precision.

### Shallow neural networks: general case
We define a general equation for a shallow neural network $\mathbf y = \mathbf f[\mathbf x, \mathbf \phi]$ that maps a multi-dimensional input $\mathbf x \in \mathbb R^{D_i}$ to a multi-dimensional output $\mathbf y \in \mathbb R^{D_o}$ using $\mathbf h \in \mathbb R^D$ hidden units. Each hidden unit is computed as:

$$
\tag{3}
h_d = a \left[ \theta_{d0} + \sum\limits_{i=1}^{D_i} \theta_{di}x_i \right],
$$

and these are combined linearly to create the output:

$$
\tag{4}
y_j = \phi_{j0} + \sum\limits_{d=1}^D \phi_{jd}h_d,
$$

where $a$ is a nonlinear activation function.

The activation function permits the model to describe nonlinear relations between input and output, and as such, it must be nonlinear itself. With no activation function, or a linear activation function, the overall mapping from input to output would be restricted to linear.

Many different activation functions have been tried, but the most common choice is the ReLU, which has the merit of being easily interpretable. With ReLU activations, the network divides the input space into convex polytopes defined by the intersections of hyperplanes computed by the "joints" in the ReLU functions. Each convex polytope contains a different linear function. The polytopes are the same for each output, but the linear functions they contain can differ.
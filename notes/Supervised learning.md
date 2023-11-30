We aim to build a model that takes an input $\mathbf x$ and produces a prediction $\mathbf y$. The model also contains parameters $\phi$, so:

$$
\mathbf y = f[\mathbf x, \phi].
$$
When we talk about *learning* or *training* a model we talk about finding parameters $\phi$ that make sensible output predictions from the input. We learn this from using a *training dataset* of $I$ pairs of input and output examples $\{\mathbf x_i, \mathbf y_i\}$.

To determine how well a model is performing we define a scalar value, called *loss* $L$, that summarizes how poor a model is performing. We define the loss as a function $L[\phi]$ of the parameters. When we train the model we are seeking parameters $\hat \phi$ that minimizes the loss function:
$$
\hat \phi = \underset{\phi}{\text{argmin}} \left[ L[\phi] \right].
$$
After training a model, we assess its performance using a separate *test dataset* to see how well it generalizes to examples that it didn't observe during training.

## Linear regression example (1D)
A 1D linear regression model described the relationship between the input and output in a straight line:
$$
y = \phi_0 + \phi_1 x
$$
This model has two parameters: $\phi_0$ and $\phi_1$.

### Loss
For this model we use the *least-squares loss* function:
$$
L[\phi] = \sum^I_{n=1}(\phi_0 + \phi_1 x_i - y_i)^2.
$$
Shallow neural networks are functions that map multivariate inputs $\mathbf x$ to multivariate outputs $\mathbf y$, however before we do that we will start with a simple network that maps a scalar value $x$ to a scalar value $y$ that has ten parameters $\phi = \{\phi_0, \phi_1, \phi_2, \phi_3, \theta_{10}, \theta_{11}, \theta_{20}, \theta_{21}, \theta_{30}, \theta_{31}\}$, giving the model:

$$
\tag 1 y = \phi_0 + \phi_1 a[\theta_{10} + \theta_{11}x]
	      + \phi_2 a[\theta_{20} + \theta_{21}x]
	      + \phi_3 a[\theta_{30} + \theta_{31}x]
$$

We can break this down into three parts: first we compute three linear functions of the input data ($\theta_{10} + \theta_{11}x$, $\theta_{20} + \theta_{21}x$, $\theta_{30} + \theta_{31}x$). Second, we pass the results through an *activation function* $a[*]$. Finally, we weight the three resulting activations with $\phi_1, \phi_2, \phi_3$, sum them and add an offset $\phi_o$.
## ReLU
The *rectified linear unit* function is an activation function, that essentially clamps all the values between $[0, \infty)$. It is defined as $\text{ReLU}[z]$ which is $z$ if $z \geq 0$ and $0$ otherwise.

## Neural network intuition
Equation (1) represent a family of continuous piecewise linear functions with up to four linear regions Here we break it down to understand why. First we split the function into two parts:

$$
\begin{align}
h_1 &= a[\theta_{10} + \theta_{11}x] \\
h_2 &= a[\theta_{20} + \theta_{21}x] \\
h_3 &= a[\theta_{30} + \theta_{31}x] \\
\end{align}
$$

where we refer to $h_1, h_2$ and $h_3$ as *hidden units*. Second, we compute the output by combining these hidden units with a linear function:

$$
y = \phi_0 + \phi_1 h_1 + \phi_2 h_2 + \phi_3 h_3
$$

Each hidden unit contains a linear function $\theta_{*0} + \theta_{*1}x$ of the input, and that line is clipped by the ReLU function below zero. The position where the three lines cross zero become the three "joins" in the final output. The three clipped lines are then weighted by $\phi_1, \phi_2$ and $\phi_3$, respectively. Finally the offset $\phi_0$ is added, which controls the overall height of the final function.

Each hidden unit contributes one "joint" to the function, so with three hidden units there can be four linear regions. However, only three of the slopes of these regions are independent; the fourth is either zero or is a sum of slopes from the other regions.

See `shallow-neural-networks/3by3.ipynb` for more examples.
## Universal approximation theorem
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

## Multivariate inputs and output
The universal approximation theorem also holds for the more general case where the network maps multivariate inputs $\mathbf x = [x_1, x_2, ..., x_{D_i}]$ to multivariate output predictions $\mathbf y = [y_1, y_2, ..., y_{D_o}]$.

### Visualizing multivariate outputs
A network with a scalar input $x$, four hidden units $h_1, h_2, h_3,$ and $h_4$, and a 2D multivariate output $\mathbf y = [y_1, y_2]$ would be defined as:

$$
\begin{align}
h_1 &= a[\theta_{10} + \theta_{11}x] \\
h_2 &= a[\theta_{20} + \theta_{21}x] \\
h_3 &= a[\theta_{30} + \theta_{31}x] \\
h_4 &= a[\theta_{40} + \theta_{41}x], \\
\end{align}
$$

and

$$
\begin{align}
y_1 &= \phi_{10} + \phi_{11}h_1 + \phi_{12}h_2 + \phi_{13}h_3 + \phi_{14}h_4 \\
y_2 &= \phi_{20} + \phi_{21}h_1 + \phi_{22}h_2 + \phi_{23}h_3 + \phi_{24}h_4. \\
\end{align}
$$

See `shallow-neural-networks/multivariate-inputs-and-outputs.ipynb` for more examples.



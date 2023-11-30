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
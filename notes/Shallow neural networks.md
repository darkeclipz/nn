Shallow neural networks are functions that map multivariate inputs $\mathbf x$ to multivariate outputs $\mathbf y$, however before we do that we will start with a simple network that maps a scalar value $x$ to a scalar value $y$ that has ten parameters $\phi = \{\phi_0, \phi_1, \phi_2, \phi_3, \theta_{10}, \theta_{11}, \theta_{20}, \theta_{21}, \theta_{30}, \theta_{31}\}$.

## ReLU
The *rectified linear unit* function is an activation function, that essentially clamps all the values between $[0, \infty)$. It is defined as $\text{ReLU}[z]$ which is $z$ if $z \geq 0$ and $0$ otherwise.
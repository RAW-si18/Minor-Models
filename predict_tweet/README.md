Implementing logistic regression for sentiment analysis on tweets.
Given a tweet, will decide if it has a positive sentiment or a negative one.

### Formula Used
$$ h(z) = \frac{1}{1+\exp^{-z}}$$
$$z = \theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2 + ... \theta_N x_N$$
$$ \theta_i \: is \: Weights \: and  \: x_i  \: is  \: Parameter $$
$$ z \: is \: logits $$

#### Cost Function
$$J(\theta) = -\frac{1}{m} \sum_{i=1}^m y^{(i)}\log (h(z(\theta)^{(i)})) + (1-y^{(i)})\log (1-h(z(\theta)^{(i)}))$$
* $m$ is the number of training examples
* $y^{(i)}$ is the actual label of training example 'i'.
* $h(z^{(i)})$ is the model's prediction for the training example 'i'.

Loss function (single training):
$$ Loss = -1 \times \left( y^{(i)}\log (h(z(\theta)^{(i)})) + (1-y^{(i)})\log (1-h(z(\theta)^{(i)})) \right)$$

* 0 <= $h$ <= 1, so the loss = -ve. Multiply the factor of -1.
* If model predicts 1 i.e. $h(z(\theta)) = 1$ & label 'y' is also 1, the loss = 0. 
* Similarly, $h(z(\theta)) = 0$ &  label is also 0, loss = 0. 
* If $h(z(\theta)) = 0.9999$ & label is 0, Then $-1 \times (0$ + $(1 - 0) \times log(1 - 0.9999)) \approx 9.2$ The closer the model prediction gets to 1, the larger the loss.

#### Update the weights

$$\mathbf{\theta} = \begin{pmatrix}
\theta_0
\\
\theta_1
\\ 
\theta_2 
\\ 
\vdots
\\ 
\theta_n
\end{pmatrix}$$
i.e. n terms and 1 bias $\theta_0$

Weight vector $\theta$, will apply gradient descent to iteratively improve model's predictions.  
Gradient of the cost function $J$ wrt. weights $\theta_j$ is:

$$\nabla_{\theta_j}J(\theta) = \frac{1}{m} \sum_{i=1}^m(h^{(i)}-y^{(i)})x^{(i)}_j$$
* 'i' is the index across all 'm' training examples.
* 'j' is the index of the weight $\theta_j$, so $x^{(i)}_j$ is the feature associated with weight $\theta_j$

* To update the weight $\theta_j$, Adjust it by subtracting a fraction of the gradient determined by $\alpha$:
$$\theta_j = \theta_j - (\alpha \times \nabla_{\theta_j}J(\theta)) $$
* The learning rate $\alpha$ i.e. how big a single update will be.

<a name='ex-2'></a>
### Gradient Descent

* 'num_iters" is the number of times entire training set will be used.
* For each iteration, Calculate the cost function using entire training set m.
* Update all the weights $\theta_i$ in the column vector:  
$$\mathbf{\theta} = \begin{pmatrix}
\theta_0
\\
\theta_1
\\ 
\theta_2 
\\ 
\vdots
\\ 
\theta_n
\end{pmatrix}$$
* $\mathbf{\theta}$ has dim (n+1, 1), where 'n' = number of features, & one  bias term $\theta_0$ (Value $\mathbf{x_0}$ is 1).
*  'logits'  $z = \mathbf{x}.\mathbf{\theta}$
    * $\mathbf{x}$ has dim (m, n+1) 
    * $\mathbf{\theta}$: has dim (n+1, 1)
    * $\mathbf{z}$: has dim (m, 1)
* Prediction $h(z) = sigmoid(z)$, and has dim (m,1).
* The cost function:
$$J = \frac{-1}{m} \times \left(\mathbf{y}^T \cdot log(\mathbf{h}) + \mathbf{(1-y)}^T \cdot log(\mathbf{1-h}) \right)$$
* Update Theta :
$$\mathbf{\theta} = \mathbf{\theta} - \frac{\alpha}{m} \times \left( \mathbf{x}^T \cdot \left( \mathbf{h-y} \right) \right)$$
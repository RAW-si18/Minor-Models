# Basic English to French Translator

### Naive Machine Translation (NMT)

### Translation as Linear Transformation of Embeddings

Given dictionaries of English and French word embeddings now will create a transformation matrix `R`
* English word embedding, $\mathbf{e}$, multiply $\mathbf{eR}$ to get a new word embedding $\mathbf{f}$.
* Both $\mathbf{e}$ and $\mathbf{f}$ are row vectors.
* Compute the nearest neighbors to `f` in the french embeddings and recommend the word that is most similar to the transformed word embedding.

#### Describing translation as the minimization problem

Matrix `R` should minimizes the equation. 

$$\arg \min _{\mathbf{R}}\| \mathbf{X R} - \mathbf{Y}\|_{F}$$

#### Frobenius norm

The Frobenius norm of a matrix $A$ (assuming it is of dimension $m,n$) is defined as the square root of the sum of the absolute squares of its elements:

$$\|\mathbf{A}\|_{F} \equiv \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n}\left|a_{i j}\right|^{2}}$$

#### Actual loss function
In the real applications, the Frobenius norm loss:

$$\| \mathbf{XR} - \mathbf{Y}\|_{F}$$

is often replaced by:

$$ \frac{1}{m} \|  \mathbf{X R} - \mathbf{Y} \|_{F}^{2}$$

where $m$ is the number of examples (rows in $\mathbf{X}$).

* Same R is found when using this loss function V.S. original Frobenius norm.
* Reason: It's easier to compute the gradient of the squared Frobenius.
* Reason: Dividing by $m$ for the average loss per embedding in training set.

#### Computing the loss
* The loss function formula is:
$$ L(X, Y, R)=\frac{1}{m}\sum_{i=1}^{m} \sum_{j=1}^{n}\left( a_{i j} \right)^{2}$$

where $a_{i j}$ is value in $i$ th row and $j$ th column of the matrix $\mathbf{XR}-\mathbf{Y}$.

#### Computing the gradient of loss wrt. transform matrix R

* Calculate the gradient of the loss with respect to transform matrix `R`.
* The gradient is a matrix that encodes how much a small change in `R`
affect the change in the loss function.
* The gradient gives us the direction in which we should decrease `R`
to minimize the loss.
* $m$ is the number of training examples (number of rows in $X$).
* The formula for the gradient of the loss function $𝐿(𝑋,𝑌,𝑅)$ is:

$$\frac{d}{dR}𝐿(𝑋,𝑌,𝑅)=\frac{d}{dR}\Big(\frac{1}{m}\| X R -Y\|_{F}^{2}\Big) = \frac{2}{m}X^{T} (X R - Y)$$

##### Gradient Descent

Gradient descent is an iterative algorithm which is used in searching for the optimum of the function. 
* Gradient descent uses that information to iteratively change matrix `R` until we reach a point where the loss is minimized. 

https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html

Pseudocode:
1. Calculate gradient $g$ of the loss with respect to the matrix $R$.
2. Update $R$ with the formula:
$$R_{\text{new}}= R_{\text{old}}-\alpha g$$

Where $\alpha$ is the learning rate, which is a scalar.

#### Learning Rate

* The learning rate or "step size" $\alpha$ is a coefficient which decides how much we want to change $R$ in each step.
* If we change $R$ too much, we could skip the optimum by taking too large of a step.
* If we make only small changes to $R$, we will need many steps to reach the optimum.
* Learning rate $\alpha$ is used to control those changes.
* Values of $\alpha$ are chosen depending on the problem, and we'll use `learning_rate`$=0.0003$ as the default value for our algorithm.

#### k-Nearest Neighbors Algorithm

k-Nearest neighbors algorithm # https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
* k-NN is a method which takes a vector as input and finds the other vectors in the dataset that are closest to it. 
* The 'k' is the number of "nearest neighbors" to find (e.g. k=2 finds the closest two neighbors).

#### Searching for the Translation Embedding
Approximating the translation function from English to French embeddings by a linear transformation matrix $\mathbf{R}$, most of the time we won't get the exact embedding of a French word when we transform embedding $\mathbf{e}$ of some particular English word into the French embedding space. 
* This is where $k$-NN becomes really useful! By using $1$-NN with $\mathbf{eR}$ as input, we can search for an embedding $\mathbf{f}$ (as a row) in the matrix $\mathbf{Y}$ which is the closest to the transformed vector $\mathbf{eR}$

#### Cosine Similarity
Cosine similarity between vectors $u$ and $v$ calculated as the cosine of the angle between them.
The formula is 

$$\cos(u,v)=\frac{u\cdot v}{\left\|u\right\|\left\|v\right\|}$$

* $\cos(u,v)$ = $1$ when $u$ and $v$ lie on the same line and have the same direction.
* $\cos(u,v)$ is $-1$ when they have exactly opposite directions.
* $\cos(u,v)$ is $0$ when the vectors are orthogonal (perpendicular) to each other.

#### Distance and similarity are pretty much opposite things.
* We can obtain distance metric from cosine similarity, but the cosine similarity can't be used directly as the distance metric. 
* When the cosine similarity increases (towards $1$), the "distance" between the two vectors decreases (towards $0$). 
* We can define the cosine distance between $u$ and $v$ as
$$d_{\text{cos}}(u,v)=1-\cos(u,v)$$

#### Accuracy

$$\text{accuracy}=\frac{\#(\text{correct predictions})}{\#(\text{total predictions})}$$


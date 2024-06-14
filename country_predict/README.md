# Country Prediction by their Capital
#### Predict relationships among words

Using the word embeddings to predict relationships among words:
* The first two are related to each other.
* As an example, "Athens is to Greece as Bangkok is to ______"?

A similar analogy: King - Man + Woman = Queen

* Compute the cosine similarity metric or the Euclidean distance.

### Cosine Similarity:

$$\cos (\theta)=\frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|\|\mathbf{B}\|}=\frac{\sum_{i=1}^{n} A_{i} B_{i}}{\sqrt{\sum_{i=1}^{n} A_{i}^{2}} \sqrt{\sum_{i=1}^{n} B_{i}^{2}}}$$

$A$ and $B$ represent the word vectors and $A_i$ or $B_i$ represent index i of that vector. If A and B are identical, you will get $cos(\theta) = 1$.
* Otherwise, if they are the total opposite, meaning, $A= -B$, then you would get $cos(\theta) = -1$.
* If you get $cos(\theta) =0$, that means that they are orthogonal (or perpendicular).
* Numbers between 0 and 1 indicate a similarity score.
* Numbers between -1 and 0 indicate a dissimilarity score.

### Euclidean Distance

Computes the similarity between two vectors using the Euclidean distance.
Euclidean distance is defined as:

$$ \begin{aligned} d(\mathbf{A}, \mathbf{B})=d(\mathbf{B}, \mathbf{A}) &=\sqrt{\left(A_{1}-B_{1}\right)^{2}+\left(A_{2}-B_{2}\right)^{2}+\cdots+\left(A_{n}-B_{n}\right)^{2}} \\ &=\sqrt{\sum_{i=1}^{n}\left(A_{i}-B_{i}\right)^{2}} \end{aligned}$$

* $n$ is the number of elements in the vector
* $A$ and $B$ are the corresponding word vectors. 
* The more similar the words, the more likely the Euclidean distance will be close to 0. 

#### Accuracy
\[ \text{Accuracy} = \frac{\text{Correct \# of predictions}}{\text{Total \# of predictions}} \]

## Plotting the vectors using PCA

*principal component analysis* (PCA) https://en.wikipedia.org/wiki/Principal_component_analysis
300-Dimensional space in this project
It is impossible to visualize results in such high dimensional spaces.

PCA is a method that projects our vectors in a space of reduced dimension, while keeping the maximum information about the original vectors in their reduced counterparts. In this case, by *maximum infomation* mean that the Euclidean distance between the original vectors and their projected siblings is minimal. Hence vectors that were originally close in the embeddings dictionary, will produce lower dimensional vectors that are still close to each other.

#### PCA

1. Mean normalize the data
2. Compute the covariance matrix of your data ($\Sigma$). 
3. Compute the eigenvectors and the eigenvalues of your covariance matrix
4. Multiply the first K eigenvectors by your normalized data.

### compute_pca

* The word vectors are of dimension 300.
* Use PCA to change the 300 dimensions to `n_components` dimensions. 
* The new matrix should be of dimension `m, n_components`. 
* First de-mean the data
* Get the eigenvalues using `linalg.eigh`.  Use 'eigh' rather than 'eig' since R is symmetric.  The performance gain when using eigh instead of eig is substantial.
* Sort the eigenvectors and eigenvalues by decreasing order of the eigenvalues.
* Get a subset of the eigenvectors (choose how many principle components you want to use using n_components).
* Return the new transformation of the data by multiplying the eigenvectors with the original data.
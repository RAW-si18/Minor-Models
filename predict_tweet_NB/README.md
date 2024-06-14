# Tweet Sentiment using Naive Bayes
Naive bayes is an algorithm that could be used for sentiment analysis. It takes a short time to train and also has a short prediction time.

$P(D_{pos})$ is the probability that the document is positive.
$P(D_{neg})$ is the probability that the document is negative.

$$P(D_{pos}) = \frac{D_{pos}}{D}$$

$$P(D_{neg}) = \frac{D_{neg}}{D}$$

Where,
$D$ = number of tweet
$D_{pos}$ = number of pos tweets 
$D_{neg}$ = number of neg tweets

#### Prior and Logprior

Prior probability represents if we had no specific information and blindly picked a tweet out of the population set, what is the probability that it will be positive versus that it will be negative? That is the "prior".

prior = $\frac{P(D_{pos})}{P(D_{neg})}$

Log of the prior to rescale:

$$\text{logprior} = log \left( \frac{P(D_{pos})}{P(D_{neg})} \right) = log \left( \frac{D_{pos}}{D_{neg}} \right)$$


$$\text{logprior} = \log (P(D_{pos})) - \log (P(D_{neg})) = \log (D_{pos}) - \log (D_{neg})$$

#### Positive and Negative Probability of a Word

- $freq_{pos}$ and $freq_{neg}$ are the freqs of that specific word in the positive or negative class.
- $N_{pos}$ and $N_{neg}$ are number of positive and negative words for all tweets respectively.
- $V$ is the number of unique words in the entire set of dataset.

$$ P(W_{pos}) = \frac{freq_{pos} + 1}{N_{pos} + V}$$
$$ P(W_{neg}) = \frac{freq_{neg} + 1}{N_{neg} + V}$$

Added "+1" in the numerator for additive smoothing.  https://en.wikipedia.org/wiki/Additive_smoothing

$$\text{loglikelihood} = \log \left(\frac{P(W_{pos})}{P(W_{neg})} \right)$$

$$ p = logprior + \sum_i^N (loglikelihood_i)$$


#### Just to check more positiveness or negativeness

$$ ratio = \frac{\text{pos\_words} + 1}{\text{neg\_words} + 1} $$

<table>
    <tr>
        <td>
            <b>Words</b>
        </td>
        <td>
        Positive word count
        </td>
         <td>
        Negative Word Count
        </td>
  </tr>
    <tr>
        <td>
        glad
        </td>
         <td>
        41
        </td>
    <td>
        2
        </td>
  </tr>
    <tr>
        <td>
        happy
        </td>
         <td>
        57
        </td>
    <td>
        4
        </td>
  </tr>
    <tr>
        <td>
        :(
        </td>
         <td>
        1
        </td>
    <td>
        3663
        </td>
  </tr>
    <tr>
        <td>
        :-(
        </td>
         <td>
        0
        </td>
    <td>
        378
        </td>
  </tr>
</table>

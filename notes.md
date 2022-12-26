## Intro to Machine Learning (ud120)

**Supervised learning** is when you performs classifications on a dataset that already has labeled examples.

Examples:

- Given a user's music choices, predict whether they like or dislike a new song based on features of a song.
- From an album of tagged photos, recognize someone in a photo.

Non-examples:

- Analyze bank data for weird-looking transactions.
- Cluster students in a course based on learning preferences.

## Scatter plots

**Decision Surface** in a scatter plot is the boundary between multiple classes of data.

ML algorithms define a decision surface which helps disambiguate data that fall in the "unclear" region.

### Exercise: intro to sklearn

The provided code is... rough. I eventually wrangled it into shape. So far the concepts are simple, but will need to spend time on numpy.

## Transition to Naive Bayes

Example:

P(C) = probability of having cancer

P(C) = 0.01

90% it is positive if you have C <-- sensitivity

90% it is negative if you don't have C <-- specificity

**Given that you test positive**, what is the probability that you have cancer? (It's about 8.33%)

```
(Prior probability) * (evidence) = (posterior probability)

P(C) = 0.01
P(-C) = 0.99
P(+|C)  = 0.9 <-- sensitivity
P(-|-C) = 0.9 <-- specificity
P(-|C)  = 0.1
P(+|-C) = 0.1

Joint
P(C|+)  = P(C)  * P(+|C)  = 0.009
P(-C|+) = P(-C) * P(+|-C) = 0.099

P(+) = P(C|+) + P(-C|+) = 0.108 <-- normalization constant

P(C|+) = P(+|C) * P(C) / P(+)
P(C|+) = 0.9 * 0.01 / 0.108 = 0.0833

P(-C|+) = P(+|-C) * P(-C) / P(+)
P(-C|+) = 0.1 * 0.99 / 0.108 = 0.9167
```

A bit obtuse in its explanation, but it makes sense that we can take advatage of the binary (yes/no) nature of the question at hand in order to derive the result.

## Support Vector Machines (SVM)

- "machine" == "algorithm"
- defines a decision boundary AKA "hyperplane"
- **margin** is the distance between the hyperplane and the closest data point
- a good hyperplane is one that maximizes the margin
- sometimes the data is not linearly separable, so we can use a **kernel trick** to transform the data into a higher dimension
- VSM parameters include the kernel type, C, and gamma
  - C controls the tradeoff between **smooth decision boundary** and **classifying training points correctly** (high C = low margin)
  - gamma defines how far the influence of a single training example reaches (low gamma = far reach)
- VSM parameters can be adjusted to prevent **overfitting**

## Decision Trees

- **entropy** controls how a DT decides where to split the data
- entropy is a measure of **impurity** in a bunch of examples

<!-- prettier-ignore -->
$$ H(x) = -\sum_{i=1}^n p(x_i) \log_2 p(x_i) $$

```
information gain = entropy before split - weighted average of entropy after split
```

## Choose your own adventure

| strategy            | accuracy |
| ------------------- | -------- |
| naive bayes         | 0.9733   |
| svm                 | 0.9960   |
| decision trees      | 0.9778   |
| k-nearest neighbors | 0.9795   |
| ada boost           | 0.9505   |

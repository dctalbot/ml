# machine learning

```
make .venv
```

## Resources

Thank you to the following resources and people:

- [Udacity UD120 with Sebastian Thrun and Katie Malone](https://www.youtube.com/watch?v=ICKBWIkfeJ8&list=PLAwxTw4SYaPkQXg8TkVdIvYv4HfLG7SiH)
- [StatQuest with Josh Starmer](https://www.youtube.com/@statquest)
- The Elements of Statistical Learning
- [scikit learn folks](https://scikit-learn.org/stable/about.html#authors)

## Notes

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

## Naive Bayes

Example:

P(C) = probability of having cancer

P(C) = 0.01

90% it is positive if you have C <-- sensitivity

90% it is negative if you don't have C <-- specificity

**Given that you test positive**, what is the probability that you have cancer? (It's about 8.33%)

```math
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

> Trees have one aspect that prevent them from being the ideal tool for predictive learning, namely inaccuracy.

## Empirical comparison of classification algorithms

Enron email classification result:

| strategy            | accuracy |
| ------------------- | -------- |
| naive bayes         | 0.9733   |
| svm                 | 0.9960   |
| decision trees      | 0.9778   |
| k-nearest neighbors | 0.9795   |
| ada boost           | 0.9505   |
| random forest       | 0.9966   |

## Continuous Supervised Learning

- Up until now in this course, we've been dealing with classification problems in which the output is a **discrete** value.
- **regression** is a supervised learning algorithm that predicts a continuous value
- We want to minimize the sum of the squared errors (SSE) between the predicted and actual values
- 2 algorithms for minimizing SSE:
  - **ordinary least squares** is a closed-form solution that finds the best fit line
  - **gradient descent** is an iterative algorithm that starts with a random guess and iterates until it finds the best fit
- Beware: One short coming of SSE is that as you inspect more data, the SSE will almost always increase simply by virtue of having more data points. This is why we use **R-squared** to measure the fitness of a regression line.
- [**R-squared**](https://www.youtube.com/watch?v=bMccdk8EdGo) is a measure of how well the regression line fits the data (1 = perfect fit, 0 = no fit)

## Comparing Supervised Classification & Supervised Regression

- Aspects of the Regression technique often have analogues in the Classification technique, and vice versa.

| Property           | Classification    | Regression    |
| ------------------ | ----------------- | ------------- |
| Output             | Discrete          | Continuous    |
| Result of training | Decision boundary | Best fit line |
| Evaluation metric  | Accuracy          | R-squared     |

## Multivariate Regression

- **multivariate regression** is a regression technique that uses more than one feature to predict a continuous value

## Outliers

### Outlier Rejection

```mermaid
graph LR
  Start --> A[Data]
  A[Train] --"∃ outliers"--> B[Remove 10%] --> A
  A --good enough--> D[Done]
```

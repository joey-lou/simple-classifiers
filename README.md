# Simple Classifiers

Another implementation exercise on writing common ML packages.  

## Naive Bayes
Different models available for input features: `gaussian`, `bernoulli`, `multi-nomial`
### Gaussian Normals
$p(x|y=k) = \frac{1}{\sqrt{2\pi \sigma_k^2}} e^{-(x-\mu_k)^2 / (2\sigma_k^2)}$

## Logistic Regression
Binary logistic regression uses cross entroy loss:
$L(w) = \sum_i y_i ln(\frac{1}{s(w^T x_i)})+(1-y_i)ln(\frac{1}{1- s(w^T x_i)})$

Multi-class can use either `ovr` or `multi-nomial`


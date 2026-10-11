# Linear Regression

`LinearRegression` is a supervised learning model for predicting a continuous target variable.

DeepKit's implementation supports four optimization strategies:

- `normal` — Normal Equation
- `batch` — Batch Gradient Descent
- `stochastic` — Stochastic Gradient Descent
- `mini-batch` — Mini-Batch Gradient Descent

---

## Import

```python
from deepkit.linear_model import LinearRegression
```

> Adjust the import path if `LinearRegression` is exposed from a different DeepKit module.

---

## Basic Usage

```python
import numpy as np
from deepkit.linear_model import LinearRegression

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5]
])

y = np.array([
    2,
    4,
    6,
    8,
    10
])

model = LinearRegression()

model.fit(X, y)

predictions = model.predict(X)

print(predictions)
```

`fit()` trains the model and returns the model itself, so method chaining is possible:

```python
model = LinearRegression().fit(X, y)
predictions = model.predict(X)
```

---

# Model

For multiple linear regression, DeepKit models the target as:

$$
\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n
$$

where:

- $\hat{y}$ is the predicted value
- $\theta_0$ is the intercept
- $\theta_1,\ldots,\theta_n$ are the feature coefficients
- $x_1,\ldots,x_n$ are the input features

Internally, DeepKit adds a column of ones to the feature matrix so the intercept can be learned as part of the parameter vector.

---

# Constructor

```python
LinearRegression(
    solver='normal',
    alpha=0.01,
    epochs=200,
    batch_size=32
)
```

## Parameters

### `solver`

Defines the optimization method.

Available values:

```python
"normal"
"batch"
"stochastic"
"mini-batch"
```

Default:

```python
solver="normal"
```

### `alpha`

Learning rate used by the gradient-based solvers.

Default:

```python
alpha=0.01
```

It controls the size of each parameter update.

A value that is too large can make training unstable, while a value that is too small can make training slow.

`alpha` is not used by the Normal Equation solver.

### `epochs`

Number of complete passes through the training data for gradient-based solvers.

Default:

```python
epochs=200
```

`epochs` is not used by the Normal Equation solver.

### `batch_size`

Number of samples used in each update for Mini-Batch Gradient Descent.

Default:

```python
batch_size=32
```

This parameter is relevant to:

```python
solver="mini-batch"
```

---

# Training Methods

## 1. Normal Equation

The Normal Equation provides a closed-form solution:

$$
\theta = X^+y
$$

where $X^+$ is the Moore-Penrose pseudoinverse of $X$.

DeepKit uses:

```python
np.linalg.pinv(X) @ y
```

The pseudoinverse makes the implementation more robust when the feature matrix is singular or not directly invertible.

### Example

```python
model = LinearRegression(solver="normal")

model.fit(X, y)
```

The Normal Equation does not require:

- `alpha`
- `epochs`
- `batch_size`

---

# 2. Batch Gradient Descent

Batch Gradient Descent calculates the gradient using the entire training dataset at every iteration.

The Mean Squared Error cost used by DeepKit is:

```math
J(\theta)
=
\frac{1}{2m}
\sum_{i=1}^{m}
\left(\hat{y}^{(i)} - y^{(i)}\right)^2
```

The gradient is:

```math
\nabla_{\theta} J
=
\frac{1}{m} X^T (X\theta - y)
```

The parameters are updated using:

```math
\theta
\leftarrow
\theta - \alpha \nabla_{\theta} J
```

### Example

```python
model = LinearRegression(
    solver="batch",
    alpha=0.01,
    epochs=200
)

model.fit(X, y)
```

---

# 3. Stochastic Gradient Descent

Stochastic Gradient Descent updates the parameters using one training sample at a time.

For each sample:

```text
calculate gradient
        ↓
update theta
        ↓
move to next sample
```

### Example

```python
model = LinearRegression(
    solver="stochastic",
    alpha=0.01,
    epochs=200
)

model.fit(X, y)
```

SGD can be useful when working with larger datasets because each update only uses one sample.

---

# 4. Mini-Batch Gradient Descent

Mini-Batch Gradient Descent divides the training data into smaller batches.

For example, with:

```python
batch_size=32
```

the model processes up to 32 samples per update.

### Example

```python
model = LinearRegression(
    solver="mini-batch",
    alpha=0.01,
    epochs=200,
    batch_size=32
)

model.fit(X, y)
```

Mini-Batch Gradient Descent provides a compromise between Batch Gradient Descent and Stochastic Gradient Descent.

---

# Input Format

`X` must be a 2D array:

```text
(n_samples, n_features)
```

Example with 5 samples and 2 features:

```python
X = np.array([
    [1, 10],
    [2, 20],
    [3, 30],
    [4, 40],
    [5, 50]
])
```

The target `y` should contain one value for each sample:

```python
y = np.array([3, 6, 9, 12, 15])
```

Therefore:

```text
X.shape = (5, 2)
y.shape = (5,)
```

The implementation automatically converts `X` and `y` to NumPy arrays.

---

# Prediction

After fitting the model:

```python
predictions = model.predict(X)
```

Example:

```python
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print(y_pred)
```

The model checks that the number of features in the prediction data matches the number of features used during training.

---

# Model Parameters

## `intercept_`

Returns the learned intercept:

```python
model.intercept_
```

Example:

```python
print("Intercept:", model.intercept_)
```

---

## `coef_`

Returns the learned coefficients:

```python
model.coef_
```

For example, with two features:

```python
print(model.coef_)
```

might return:

```text
[1.5, 2.3]
```

---

## `theta_`

Returns the complete parameter vector, including the intercept:

```python
model.theta_
```

For example:

```text
[2.1, 1.5, 2.3]
```

The first value is the intercept and the remaining values are the feature coefficients.

---

# Model Information

## `n_features_in_`

Number of features used during training:

```python
print(model.n_features_in_)
```

---

## `n_iter_`

Number of training iterations/epochs performed:

```python
print(model.n_iter_)
```

For the Normal Equation, the value is set to:

```text
1
```

For gradient-based solvers, it corresponds to the configured number of epochs.

---

# Training History

DeepKit stores the cost calculated during gradient-based training in:

```python
model.cost_history_
```

Example:

```python
model = LinearRegression(
    solver="batch",
    alpha=0.01,
    epochs=100
)

model.fit(X, y)

print(model.cost_history_)
```

You can use this history to study how the cost changes during training.

For example, it can be plotted with Matplotlib:

```python
import matplotlib.pyplot as plt

plt.plot(model.cost_history_)
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Training Cost")
plt.show()
```

For the Normal Equation, no iterative gradient descent history is generated.

---

# R² Score

DeepKit provides an `r2_score` utility:

```python
LinearRegression.r2_score(y, y_pred)
```

Example:

```python
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

score = LinearRegression.r2_score(y, y_pred)

print("R²:", score)
```

The coefficient of determination is:

$$
R^2 =
1 -
\frac{\sum_i(y_i-\hat{y}_i)^2}
{\sum_i(y_i-\bar{y})^2}
$$

A value closer to `1` generally indicates that the predictions explain more of the variance in the target.

---

# Complete Example

```python
import numpy as np
from deepkit.linear_model import LinearRegression

# Dataset
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6]
])

y = np.array([
    2,
    4,
    6,
    8,
    10,
    12
])

# Create the model
model = LinearRegression(
    solver="batch",
    alpha=0.01,
    epochs=1000
)

# Train
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Model information
print("Predictions:", y_pred)
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("Theta:", model.theta_)
print("Features:", model.n_features_in_)
print("Iterations:", model.n_iter_)

# R²
score = LinearRegression.r2_score(y, y_pred)
print("R²:", score)

# Training history
print("Final cost:", model.cost_history_[-1])
```

---

# Choosing a Solver

| Solver | Main idea | Learning rate | Epochs | Batch size |
|---|---|---:|---:|---:|
| `normal` | Closed-form solution | ❌ | ❌ | ❌ |
| `batch` | Full dataset per update | ✅ | ✅ | ❌ |
| `stochastic` | One sample per update | ✅ | ✅ | ❌ |
| `mini-batch` | Small batch per update | ✅ | ✅ | ✅ |

For a small dataset, the Normal Equation is convenient.

For experimenting with optimization and learning behavior, use one of the gradient-based solvers.

---

# Input Validation

DeepKit validates several conditions automatically.

For example:

### `X` must be 2D

```python
X = [1, 2, 3]
```

will raise an error because the expected format is:

```text
(n_samples, n_features)
```

### `X` and `y` must have the same number of samples

```python
X.shape[0] == y.shape[0]
```

must be true.

### Training parameters

`alpha`, `epochs`, and `batch_size` must be greater than zero.

### Prediction before fitting

Calling:

```python
model.predict(X)
```

before:

```python
model.fit(X, y)
```

raises an error.

### Feature consistency

If the model was trained with 3 features, prediction data must also contain 3 features.

---

# Numerical Stability

During gradient-based training, DeepKit checks whether the learned parameters remain finite.

If numerical instability occurs, the model raises an error suggesting that you reduce the learning rate or scale your features.

For example:

```python
model = LinearRegression(
    solver="batch",
    alpha=0.0001
)
```

Feature scaling can also help gradient-based optimization.

---

# API Summary

| Attribute / Method | Description |
|---|---|
| `fit(X, y)` | Train the model |
| `predict(X)` | Generate predictions |
| `intercept_` | Learned intercept |
| `coef_` | Learned feature coefficients |
| `theta_` | Complete parameter vector |
| `n_features_in_` | Number of input features |
| `n_iter_` | Number of training iterations |
| `cost_history_` | Cost recorded during iterative training |
| `r2_score(y, y_pred)` | Calculate R² score |

---

## Notes

DeepKit's `LinearRegression` is designed to be educational as well as practical. Its multiple solvers make it possible to compare a closed-form solution with different gradient-based optimization strategies.

For gradient-based solvers, feature scaling is generally recommended, especially when features have very different numerical ranges.

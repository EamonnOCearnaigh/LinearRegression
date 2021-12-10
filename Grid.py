import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

rng = np.random.default_rng(42)

# Generate regression dataset
X, y = make_regression(
    n_samples=1000,
    n_features=1,
    noise=0.0,
    bias=0.0,
    random_state=42,
)

# Generate regression dataset
X_noisy, y_noisy = make_regression(
    n_samples=1000,
    n_features=1,
    noise=0.0,
    bias=0.0,
    random_state=42,
)

for x in range(200):
    X = np.append(X, rng.choice(X.flatten()))
    y = np.append(y, rng.choice(y.flatten()))

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

regressor = LinearRegression()
regressor.fit(X_train, y_train)  # Training the algorithm

# To retrieve the intercept:
print(regressor.intercept_)
# For retrieving the slope:
print(regressor.coef_)

y_pred = regressor.predict(X_test)

print("R2 Score:", metrics.r2_score(y_test, y_pred))
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# # df = pd.DataFrame({"Actual": y_test.flatten(), "Predicted": y_pred.flatten()})
# # df.head(25).plot(kind="bar", figsize=(16, 10))
# # plt.grid(which="major", linestyle="-", linewidth="0.5", color="green")
# # plt.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
# # plt.show()

plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color="red", linewidth=1)
plt.show()

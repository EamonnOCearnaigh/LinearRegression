import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import scipy as sp
from scipy.stats import chi2
from sklearn.covariance import MinCovDet

# Robust Mahalonibis Distance
def robust_mahalanobis_method(df):
    # Minimum covariance determinant
    rng = np.random.RandomState(0)
    real_cov = np.cov(df.values.T)
    X = rng.multivariate_normal(mean=np.mean(df, axis=0), cov=real_cov, size=506)
    cov = MinCovDet(random_state=42).fit(X)
    mcd = cov.covariance_  # robust covariance metric
    robust_mean = cov.location_  # robust mean
    inv_covmat = sp.linalg.inv(mcd)  # inverse covariance metric

    # Robust M-Distance
    x_minus_mu = df - robust_mean
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    md = np.sqrt(mahal.diagonal())

    # Flag as outlier
    outlier = []
    C = np.sqrt(
        chi2.ppf((1 - 0.001), df=df.shape[1])
    )  # degrees of freedom = number of variables
    for index, value in enumerate(md):
        if value > C:
            outlier.append(index)
        else:
            continue
    return outlier, md


rng = np.random.default_rng(42)

performance = []

counter = 0
for std in range(0, 100, 10):
    performance.append([std, []])
    for outlier_num in range(0, 5000, 100):
        # Generate regression dataset
        X, y = make_regression(
            n_samples=5000,
            n_features=1,
            noise=0.0,
            bias=0.0,
            random_state=42,
        )

        # Generate regression dataset
        X_noisy, y_noisy = make_regression(
            n_samples=5000,
            n_features=1,
            noise=std,
            bias=0.0,
            random_state=42,
        )

        X_outliers = []
        y_outliers = []
        for i in range(outlier_num):
            X_outliers = np.append(X_outliers, rng.choice(X_noisy.flatten()))
            y_outliers = np.append(y_outliers, rng.choice(y_noisy.flatten()))

        data = np.stack((np.append(X, X_outliers), np.append(y, y_outliers)), axis=1)

        df = pd.DataFrame(data)
        outliers_mahal, md = robust_mahalanobis_method(df=df)
        outliers_mahal = np.array(outliers_mahal)
        outliers_mahal = outliers_mahal[outliers_mahal > 5000]

        X_outliers_final = []
        y_outliers_final = []

        X_temp = np.append(X, X_outliers)
        y_temp = np.append(y, y_outliers)
        for i in outliers_mahal:
            X_outliers_final = np.append(X_outliers_final, X_temp[i])
            y_outliers_final = np.append(y_outliers_final, y_temp[i])

        X = np.append(X, X_outliers_final)
        y = np.append(y, y_outliers_final)

        X = X.reshape(-1, 1)
        y = y.reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

        y_pred = regressor.predict(X_test)

        performance[counter][1].append(
            [
                outlier_num,
                metrics.r2_score(y_test, y_pred),
                metrics.mean_absolute_error(y_test, y_pred),
                metrics.mean_squared_error(y_test, y_pred),
                np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
                regressor.intercept_,
                regressor.coef_,
            ]
        )

        # print("R2 Score:", metrics.r2_score(y_test, y_pred))
        # print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
        # print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
        # print(
        #     "Root Mean Squared Error:",
        #     np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
        # )
    print(counter)
    counter += 1

# df = pd.DataFrame({"Actual": y_test.flatten(), "Predicted": y_pred.flatten()})
# df.head(25).plot(kind="bar", figsize=(16, 10))
# plt.grid(which="major", linestyle="-", linewidth="0.5", color="green")
# plt.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
# plt.show()

fig, axs = plt.subplots(3, 2)
performance[0][0] = 1
for std in performance:
    stats = [[], [], [], [], [], [], []]
    for i in std[1]:
        stats[0].append(i[0])
        stats[1].append(i[1])
        stats[2].append(i[2])
        stats[3].append(i[3])
        stats[4].append(i[4])

    axs[0, 0].plot(stats[0], stats[1], linewidth=2, label="Std = {:.0f}".format(std[0]))
    axs[0, 0].set_xlabel("Number of Outliers")
    axs[0, 0].set_ylabel("R2 Score")
    axs[0, 0].legend(loc="lower right")

    axs[0, 1].plot(stats[0], stats[2], linewidth=2, label="Std = {:.0f}".format(std[0]))
    axs[0, 1].set_xlabel("Number of Outliers")
    axs[0, 1].set_ylabel("MAE")
    axs[0, 1].legend(loc="lower right")

    axs[1, 0].plot(stats[0], stats[3], linewidth=2, label="Std = {:.0f}".format(std[0]))
    axs[1, 0].set_xlabel("Number of Outliers")
    axs[1, 0].set_ylabel("MSE")
    axs[1, 0].legend(loc="lower right")

    axs[1, 1].plot(stats[0], stats[4], linewidth=2, label="Std = {:.0f}".format(std[0]))
    axs[1, 1].set_xlabel("Number of Outliers")
    axs[1, 1].set_ylabel("RMSE")
    axs[1, 1].legend(loc="lower right")

    axs[2, 0].plot(stats[0], stats[4], linewidth=2, label="Std = {:.0f}".format(std[0]))
    axs[2, 0].set_xlabel("Number of Outliers")
    axs[2, 0].set_ylabel("Intercept")
    axs[2, 0].legend(loc="lower right")

    axs[2, 1].plot(stats[0], stats[4], linewidth=2, label="Std = {:.0f}".format(std[0]))
    axs[2, 1].set_xlabel("Number of Outliers")
    axs[2, 1].set_ylabel("Estimated Coefficient")
    axs[2, 1].legend(loc="lower right")

# # Generate regression dataset
# X, y = make_regression(
#     n_samples=5000,
#     n_features=1,
#     noise=0.0,
#     bias=0.0,
#     random_state=42,
# )

# X_noisy, y_noisy = make_regression(
#     n_samples=5000,
#     n_features=1,
#     noise=0.0,
#     bias=0.0,
#     random_state=42,
# )

# # X_outliers = []
# # y_outliers = []
# # for i in range(1000):
# #     X_outliers = np.append(X_outliers, rng.choice(X_noisy.flatten()))
# #     y_outliers = np.append(y_outliers, rng.choice(y_noisy.flatten()))

# data = np.stack((np.append(X, X_noisy), np.append(y, y_noisy)), axis=1)

# df = pd.DataFrame(data)
# outliers_mahal, md = robust_mahalanobis_method(df=df)
# outliers_mahal = np.array(outliers_mahal)
# outliers_mahal = outliers_mahal[outliers_mahal > 5000]

# X_outliers_final = []
# y_outliers_final = []

# X_temp = np.append(X, X_noisy)
# y_temp = np.append(y, y_noisy)
# for i in outliers_mahal:
#     X_outliers_final = np.append(X_outliers_final, X_temp[i])
#     y_outliers_final = np.append(y_outliers_final, y_temp[i])

# X = np.append(X, X_outliers_final)
# y = np.append(y, y_outliers_final)

# X = X.reshape(-1, 1)
# y = y.reshape(-1, 1)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# regressor = LinearRegression()
# regressor.fit(X_train, y_train)

# y_pred = regressor.predict(X_test)

# fig, ax = plt.subplots()
# ax.scatter(X, y, label="Original Data")
# ax.scatter(X_outliers_final, y_outliers_final, c="Purple", label="Outliers")
# ax.plot(X_test, y_pred, color="Red", linewidth=2, label="Intercept")
# ax.annotate(
#     "R2 Score = {:.3f}".format(metrics.r2_score(y_test, y_pred)),
#     xy=(0.015, 0.98),
#     xycoords="axes fraction",
#     verticalalignment="top",
#     horizontalalignment="left",
# )
# ax.annotate(
#     "Mean Absolute Error = {:.3f}\n".format(
#         metrics.mean_absolute_error(y_test, y_pred)
#     ),
#     xy=(0.015, 0.935),
#     xycoords="axes fraction",
#     verticalalignment="top",
#     horizontalalignment="left",
# )
# ax.annotate(
#     "Mean Squared Error = {:.3f}\n".format(metrics.mean_squared_error(y_test, y_pred)),
#     xy=(0.015, 0.886),
#     xycoords="axes fraction",
#     verticalalignment="top",
#     horizontalalignment="left",
# )
# ax.annotate(
#     "Root Mean Squared Error = {:.3f}".format(
#         np.sqrt(metrics.mean_squared_error(y_test, y_pred))
#     ),
#     xy=(0.015, 0.84),
#     xycoords="axes fraction",
#     verticalalignment="top",
#     horizontalalignment="left",
# )
# ax.annotate(
#     "Intercept = {:.3f}".format(regressor.intercept_[0]),
#     xy=(0.015, 0.795),
#     xycoords="axes fraction",
#     verticalalignment="top",
#     horizontalalignment="left",
# )
# ax.annotate(
#     "Estimated Coefficient = {:.3f}".format(regressor.coef_[0][0]),
#     xy=(0.015, 0.75),
#     xycoords="axes fraction",
#     verticalalignment="top",
#     horizontalalignment="left",
# )

plt.show()

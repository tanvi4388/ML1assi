import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


X, y = make_regression(n_samples=200, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

y_pred = poly_reg.predict(X_test_poly)


print("Polynomial Regression R²:", r2_score(y_test, y_pred))
print("Polynomial Regression MSE:", mean_squared_error(y_test, y_pred))


plt.scatter(X, y, color='gray', alpha=0.6, label='Data')
plt.scatter(X_test, y_pred, color='red', s=10, label='Polynomial Fit')
plt.legend()
plt.title("Polynomial Regression (Degree=3)")
plt.show()

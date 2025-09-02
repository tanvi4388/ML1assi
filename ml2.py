import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
y = np.array([25, 34, 44, 50, 61, 67, 74, 80, 95])
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_poly_pred, color='green', label='Polynomial Fit')
plt.title('Polynomial Regression (Degree 2)')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.legend()
plt.grid(True)
plt.show()
print("Polynomial Regression R² Score:", r2_score(y, y_poly_pred))




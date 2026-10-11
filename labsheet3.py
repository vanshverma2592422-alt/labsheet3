import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.datasets import fetch_california_housing

DATA_FILE = "data.csv"
TARGET_COLUMN = "Price"
TEST_SIZE = 0.2
RANDOM_STATE = 42


# Program 1: Load a regression dataset using Pandas
df = pd.read_csv(DATA_FILE)


# Program 2: Display the first and last five records
df = pd.read_csv(DATA_FILE)
first_five = df.head()
last_five = df.tail()


# Program 3: Explore dataset information and descriptive statistics
df = pd.read_csv(DATA_FILE)
df.info()
statistics = df.describe()


# Program 4: Identify input and output variables
df = pd.read_csv(DATA_FILE)
df = df.select_dtypes(include=np.number).dropna()
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]


# Program 5: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)


# Program 6: Implement Simple Linear Regression
df = pd.read_csv(DATA_FILE)
df = df.select_dtypes(include=np.number).dropna()
feature = df.drop(columns=[TARGET_COLUMN]).columns[0]
X_simple = df[[feature]]
y_simple = df[TARGET_COLUMN]

X_simple_train, X_simple_test, y_simple_train, y_simple_test = (
    train_test_split(
        X_simple, y_simple,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
)

simple_model = LinearRegression()


# Program 7: Train the Linear Regression model
simple_model.fit(X_simple_train, y_simple_train)


# Program 8: Predict using the trained Linear Regression model
simple_predictions = simple_model.predict(X_simple_test)


# Program 9: Visualize the Linear Regression line
sort_index = np.argsort(X_simple_test[feature].to_numpy())

plt.scatter(X_simple_test[feature], y_simple_test)
plt.plot(
    X_simple_test[feature].to_numpy()[sort_index],
    simple_predictions[sort_index]
)
plt.xlabel(feature)
plt.ylabel(TARGET_COLUMN)
plt.title("Simple Linear Regression")
plt.show()


# Program 10: Compare actual and predicted values
comparison = pd.DataFrame({
    "Actual": y_simple_test,
    "Predicted": simple_predictions
})


# Program 11: Display regression coefficient and intercept
coefficient = simple_model.coef_
intercept = simple_model.intercept_


# Program 12: Predict output for new user-defined input
new_input = pd.DataFrame({
    feature: [X_simple[feature].median()]
})
new_prediction = simple_model.predict(new_input)


# Program 13: Implement Multiple Linear Regression
df = pd.read_csv(DATA_FILE)
df = df.select_dtypes(include=np.number).dropna()
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

multiple_model = LinearRegression()


# Program 14: Train the Multiple Linear Regression model
multiple_model.fit(X_train, y_train)


# Program 15: Predict using the testing dataset
multiple_predictions = multiple_model.predict(X_test)


# Program 16: Compare actual and predicted values graphically
plt.scatter(y_test, multiple_predictions)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()


# Program 17: Analyze the effect of each independent variable
feature_effects = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": multiple_model.coef_
})


# Program 18: Implement Polynomial Regression of Degree 2
df = pd.read_csv(DATA_FILE)
df = df.select_dtypes(include=np.number).dropna()
feature = df.drop(columns=[TARGET_COLUMN]).columns[0]
X_poly = df[[feature]]
y_poly = df[TARGET_COLUMN]

X_poly_train, X_poly_test, y_poly_train, y_poly_test = (
    train_test_split(
        X_poly, y_poly,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
)

poly_model_2 = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)
poly_model_2.fit(X_poly_train, y_poly_train)


# Program 19: Implement Polynomial Regression of Degree 3
poly_model_3 = make_pipeline(
    PolynomialFeatures(degree=3),
    LinearRegression()
)
poly_model_3.fit(X_poly_train, y_poly_train)


# Program 20: Compare Linear and Polynomial Regression models
linear_model = LinearRegression()
linear_model.fit(X_poly_train, y_poly_train)

linear_predictions = linear_model.predict(X_poly_test)
poly2_predictions = poly_model_2.predict(X_poly_test)
poly3_predictions = poly_model_3.predict(X_poly_test)

model_comparison = pd.DataFrame({
    "Model": ["Linear", "Polynomial Degree 2", "Polynomial Degree 3"],
    "MAE": [
        mean_absolute_error(y_poly_test, linear_predictions),
        mean_absolute_error(y_poly_test, poly2_predictions),
        mean_absolute_error(y_poly_test, poly3_predictions)
    ],
    "MSE": [
        mean_squared_error(y_poly_test, linear_predictions),
        mean_squared_error(y_poly_test, poly2_predictions),
        mean_squared_error(y_poly_test, poly3_predictions)
    ]
})


# Program 21: Visualize Polynomial Regression curves
plot_x = np.linspace(
    X_poly[feature].min(),
    X_poly[feature].max(),
    200
).reshape(-1, 1)

plt.scatter(X_poly[feature], y_poly, alpha=0.5)
plt.plot(
    plot_x.ravel(),
    poly_model_2.predict(plot_x),
    label="Degree 2"
)
plt.plot(
    plot_x.ravel(),
    poly_model_3.predict(plot_x),
    label="Degree 3"
)
plt.xlabel(feature)
plt.ylabel(TARGET_COLUMN)
plt.legend()
plt.show()


# Program 22: Predict output using Polynomial Regression
new_poly_input = pd.DataFrame({
    feature: [X_poly[feature].median()]
})
poly_prediction = poly_model_2.predict(new_poly_input)


# Program 23: Compare prediction accuracy for different degrees
degree_results = []

for degree in range(1, 6):
    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        LinearRegression()
    )
    model.fit(X_poly_train, y_poly_train)
    predictions = model.predict(X_poly_test)

    degree_results.append({
        "Degree": degree,
        "MAE": mean_absolute_error(y_poly_test, predictions),
        "MSE": mean_squared_error(y_poly_test, predictions),
        "RMSE": np.sqrt(
            mean_squared_error(y_poly_test, predictions)
        ),
        "R2": r2_score(y_poly_test, predictions)
    })

degree_comparison = pd.DataFrame(degree_results)


# Program 24: Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, multiple_predictions)


# Program 25: Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, multiple_predictions)


# Program 26: Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(
    mean_squared_error(y_test, multiple_predictions)
)


# Program 27: Calculate R-squared (R2) Score
r2 = r2_score(y_test, multiple_predictions)


# Program 28: Compare Linear and Polynomial Regression metrics
def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mean_squared_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R2": r2_score(actual, predicted)
    }


metrics_comparison = pd.DataFrame([
    {
        "Model": "Linear Regression",
        **calculate_metrics(y_poly_test, linear_predictions)
    },
    {
        "Model": "Polynomial Degree 2",
        **calculate_metrics(y_poly_test, poly2_predictions)
    },
    {
        "Model": "Polynomial Degree 3",
        **calculate_metrics(y_poly_test, poly3_predictions)
    }
])


# Program 29: Interpret MSE and R2 values
# MSE measures average squared prediction error.
# Lower MSE generally indicates smaller prediction errors.
# R2 measures how much target variation the model explains.
# R2 closer to 1 generally indicates a better fit.


# Program 30: Visualize prediction errors using a scatter plot
prediction_errors = y_test - multiple_predictions

plt.scatter(multiple_predictions, prediction_errors)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Prediction Errors")
plt.title("Regression Prediction Errors")
plt.show()


# Program 31: Train regression model using standardized features
standardized_model = make_pipeline(
    StandardScaler(),
    LinearRegression()
)

standardized_model.fit(X_train, y_train)
standardized_predictions = standardized_model.predict(X_test)


# Program 32: Compare performance before and after feature scaling
unscaled_model = LinearRegression()
unscaled_model.fit(X_train, y_train)
unscaled_predictions = unscaled_model.predict(X_test)

scaling_comparison = pd.DataFrame([
    {
        "Model": "Before Scaling",
        **calculate_metrics(y_test, unscaled_predictions)
    },
    {
        "Model": "After Standardization",
        **calculate_metrics(y_test, standardized_predictions)
    }
])


# Program 33: Perform regression using another real-world dataset
housing = fetch_california_housing(as_frame=True)
housing_data = housing.frame.dropna()

X_housing = housing_data.drop(columns=["MedHouseVal"])
y_housing = housing_data["MedHouseVal"]

X_housing_train, X_housing_test, y_housing_train, y_housing_test = (
    train_test_split(
        X_housing, y_housing,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
)

housing_model = LinearRegression()
housing_model.fit(X_housing_train, y_housing_train)
housing_predictions = housing_model.predict(X_housing_test)


# Program 34: Save the trained regression model using Joblib
joblib.dump(standardized_model, "regression_model.joblib")


# Program 35: Load saved model and predict new data
loaded_model = joblib.load("regression_model.joblib")
new_data = X_test.iloc[[0]]
loaded_prediction = loaded_model.predict(new_data)

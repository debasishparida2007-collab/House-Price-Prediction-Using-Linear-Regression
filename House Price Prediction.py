# quick house price predictor (just experimenting tbh)

import numpy as np
from sklearn.linear_model import LinearRegression

# dataset: [area, bedrooms, age]
# yeah I just typed these manually... not from any real dataset
X_data = np.array([
    [500, 1, 10],
    [800, 2, 8],
    [1000, 2, 5],
    [1200, 3, 7],
    [1500, 3, 2]
])

# prices in lakhs
y_data = np.array([50, 80, 100, 120, 150])

# create model
model = LinearRegression()

# train it
model.fit(X_data, y_data)

# --- input part ---
# assuming user gives correct input... no checks for now

area = float(input("Enter area (sq.ft): "))
beds = int(input("Enter bedrooms: "))
house_age = int(input("Enter age of house: "))

# putting inputs together
# could've passed directly but this feels easier to read
input_vals = [[area, beds, house_age]]

# prediction
price_pred = model.predict(input_vals)

# sometimes I like storing intermediate values like this
final_output = price_pred[0]

# printing result
print("Estimated House Price:", round(final_output, 2), "lakhs")

# debug stuff (keeping it commented)
# print("coef:", model.coef_)
# print("intercept:", model.intercept_)
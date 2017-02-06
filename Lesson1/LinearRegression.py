import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#read data
dataframe = pd.read_fwf('brain_body.txt')
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

#train mdoel on data
model = LinearRegression()
model.fit(x_values, y_values)
#predict for a new input
print(model.predict([ [127], [248] ]))

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()

#read csv file
bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv")

#Linear Regression model
model = LinearRegression() #define model
mode.fit(x_values, y_values) #train model
mode.predict(21.07) #use model to predict
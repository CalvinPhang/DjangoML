import joblib
import pandas as pd
import numpy as np
from sklearn import linear_model, tree


df = pd.read_csv("waterheat.csv")

dec_tree = tree.DecisionTreeClassifier()
dec_tree = dec_tree.fit(df[['temp', 'rate', 'out']], df.heater)

joblib.dump(dec_tree, 'heater_model.pkl')


df = pd.read_csv("fancontrol.csv")

regfan = linear_model.LinearRegression()
regfan.fit(df[['temp', 'hum', 'co2']], df.fan)

joblib.dump(regfan, 'fan_model.pkl')


df = pd.read_csv("lightingcontrol.csv")

reglight = linear_model.LinearRegression()
reglight.fit(df[['light', 'occ', 'day']], df.led)

joblib.dump(reglight, 'light_model.pkl')



import pandas as pd
import numpy as np
from sklearn import linear_model
from django.conf import settings

class BaseLinearRegression:
    def __init__(self,training_data):
        self.df = pd.read_csv(settings.ML_ROOT + training_data)
        self.model = linear_model.LinearRegression()
        self.model.fit(self.df.iloc[:, 0:3], self.df.iloc[:,-1:])
        
    def predict(self, value):
        return self.model.predict([value])

water_heater_model = BaseLinearRegression('heater.csv')
fan_control_model = BaseLinearRegression('fan.csv')
light_control_model = BaseLinearRegression('light.csv')
# test_model = BaseLinearRegression('fancontrol.csv')
# print(test_model.predict([-10,10,100]))

from django.shortcuts import render
from django.views.generic import View
from django.conf import settings

from rest_framework.views import APIView
from rest_framework.response import Response

import joblib
import machinelearning

from .models import Sensor, SensorLog, Actuator, ActuatorLog



class SensorTemplateView(APIView):
    sensor_name = ""
    def get(self, request, format=None):
        sensor = Sensor.objects.get(name=self.sensor_name)
        data = {
            "value": sensor.value
        }
        return Response(data)

class ActuatorTemplateView(APIView):
    actuator_name = ""
    def get(self, request, format=None):
        actuator = Actuator.objects.get(name=self.actuator_name)
        sensors = Sensor.objects.filter(subsystem=actuator.subsystem)
        
        data = {
            "state": actuator.state
        }
        return Response(data)

class DashboardView(View):
    def get(self, request, *args, **kwargs):
        return render(request, 'home.html')

# Water Heating System
class WaterTemperatureView(SensorTemplateView):
    sensor_name = "Water Temperature"
    
class WaterUsageRateView(SensorTemplateView):
    sensor_name = "Water Usage Rate"

class OutsideTemperatureView(SensorTemplateView):
    sensor_name = "Outside Temperature"
    
class WaterHeaterView(APIView):
    def get(self, request, format=None):
        actuator = Actuator.objects.get(name="Water Heater")
        watertemp = Sensor.objects.get(name="Water Temperature")
        waterrate = Sensor.objects.get(name="Water Usage Rate")
        outtemp = Sensor.objects.get(name="Outside Temperature")
        model = joblib.load(settings.ML_ROOT + "heater_model.pkl")
        prediction = model.predict([[float(watertemp.value), float(waterrate.value), float(outtemp.value)]])
        actuator.state = int(prediction)
        actuator.save()
        data = {
            "state": int(prediction)
        }
        return Response(data)

# Fan Control System
class RoomTemperatureView(SensorTemplateView):
    sensor_name = "Room Temperature"
    
class RoomHumidityView(SensorTemplateView):
    sensor_name = "Room Humidity"
    
class RoomCO2View(SensorTemplateView):
    sensor_name = "Room CO2"
    
class VentilationFanView(ActuatorTemplateView):
    def get(self, request, format=None):
        actuator = Actuator.objects.get(name="Ventilation Fan")
        roomtemp = Sensor.objects.get(name="Room Temperature")
        roomhum = Sensor.objects.get(name="Room Humidity")
        roomco2 = Sensor.objects.get(name="Room CO2")
        model = joblib.load(settings.ML_ROOT + "fan_model.pkl")
        prediction = model.predict([[float(roomtemp.value), float(roomhum.value), float(roomco2.value)]])
        actuator.state = int(prediction)
        actuator.save()
        data = {
            "state": int(prediction)
        }
        return Response(data)
    
# Lighting Control System
class LightLevelView(SensorTemplateView):
    sensor_name = "Light Level"
    
class OccupancyView(SensorTemplateView):
    sensor_name = "Occupancy"
    
class DaylightView(SensorTemplateView):
    sensor_name = "Daylight"

class LightingSystemView(ActuatorTemplateView):
    def get(self, request, format=None):
        actuator = Actuator.objects.get(name="Lighting System")
        lightlevel = Sensor.objects.get(name="Light Level")
        occupancy = Sensor.objects.get(name="Occupancy")
        daylight = Sensor.objects.get(name="Daylight")
        model = joblib.load(settings.ML_ROOT + "light_model.pkl")
        prediction = model.predict([[float(lightlevel.value), float(occupancy.value), float(daylight.value)]])
        actuator.state = int(prediction)
        actuator.save()
        data = {
            "state": int(prediction)
        }
        return Response(data)
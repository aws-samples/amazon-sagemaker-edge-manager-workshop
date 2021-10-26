# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import time
import pandas as pd
import subprocess
import ipywidgets as widgets
import logging
import numpy as np
import os
import signal
from turbine import WindTurbine
import mqttclient
from awscrt import mqtt
import json
import threading
import logging

"""
This Class represents the simulator which is rending the simulation in notebook to present the 
wind turbines and provide utility functions to Turbine.
"""

class WindTurbineFarmSimulator(object):
    def __init__(self, n_turbines=5):
        self.n_turbines = n_turbines
        # read the raw data. This data was captured from real sensors installed in the mini Wind Turbine
        self.raw_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/dataset_wind.csv.gz'), compression="gzip", sep=',', low_memory=False).values
        
        self.mqtt_client=mqttclient.Client(client_id='simulator')
        self.mqtt_client.connect()
        
        # now create the virtual wind turbines
        self.turbines = [WindTurbine(i, self.raw_data, client=self.mqtt_client) for i in range(n_turbines)]

        self.running = False
        self.halted = False     
        
        self.feature_ids = np.array([8,9,10,7,  22, 5, 6]) # qX,qy,qz,qw  ,wind_seed_rps, rps, voltage        
        self.feature_names = np.array(['qx', 'qy', 'qz', 'qw', 'wind speed rps', 'rps', 'voltage'])

        self.dashboard = widgets.Textarea(value='\n' * self.n_turbines, disabled=True,
            layout={'border': '1px solid black', 'width': '850px', 'height': '90px'})
        
        for i in range(n_turbines):
            update_dashboard_topic = f'wind-turbine/{i}/dashboard/update'
            self.mqtt_client.subscribe(update_dashboard_topic, mqtt.QoS.AT_LEAST_ONCE, handler=self.__callback_update_dashboard__)

        

    def start(self):
        """
        Run the main application,
        kicking-off the simulator
        """
        if not self.running and not self.halted:
            self.running = True

    
    def halt(self):
        """
        Stop the turbines
        """
        if self.running:
            self.running = False
            self.halted = True
            # halt all the turbines
            for i in self.turbines: i.halt()   
            self.mqtt_client.disconnect()


    def get_num_turbines(self):
        """
        Get number of turbines
        """
        return self.n_turbines

    def show(self):
        """
        Display the dashboard for each turbine
        """
        return widgets.VBox([
            widgets.HBox([t.show() for t in self.turbines]),
            self.dashboard
        ])
    
    
    def __callback_update_dashboard__(self, topic, payload, dup, qos, retain, **kwargs):
        """
        Callback when turbine receives new data to be updated on dashboard from the inference app
        """
        turbine_id = int(topic.split("/")[1])
        json_response = json.loads(payload)
        dashboard_data = np.array(json_response).astype(object)
        self.__update_dashboard__(turbine_id, dashboard_data)


    def __update_dashboard__(self, turbine_id, data):
        """
        Updates simulator dashboard data
        """
        if not self.turbines[turbine_id].is_running(): return
        lines = self.dashboard.value.split('\n')  
        response_feature_ids = np.array([i for i in range(self.feature_ids.size)])
        features = np.mean(data[-50:,response_feature_ids], axis=0)
        tokens = ["%s: %0.3f" % (self.feature_names[i], features[i]) for i in range(len(features))]
        lines[turbine_id] = ' '.join(["Turbine: %d" % turbine_id] + tokens)        
        self.dashboard.value = '\n'.join(lines)


    def __del__(self):
        """
        Stop the simulator, which in turn will stop the turbines.
        """
        self.halt()

 

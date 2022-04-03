# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import ipywidgets as widgets
import random
import time
import json
import numpy as np
import logging
import os
import time

from awscrt import mqtt
import threading


class WindTurbine(object):
    """ Represents virtually and graphically a wind turbine
        It uses the raw data collected from a Wind Turbine in a circular buffer
        to simulate the real turbine sensors.
    """
    def __init__(self, turbine_id=0, raw_data=None, client=None):
        if raw_data is None or len(raw_data) == 0:
            raise Exception("You need to pass an array with at least one row for raw data")
        
        self.mqtt_client = client
        self.turbine_id = turbine_id # id of the turbine
        self.raw_data = raw_data # buffer with the raw sensors data
        self.raw_data_idx = random.randint(0, len(raw_data)-1)
        
        self.running = False # running status
        self.halted = False # if True you can't use this turbine anymore. create a new one.

        # components of the UI
        self.stopped_img = open('../../imgs/wind_turbine.png', 'rb').read()
        self.running_img = open('../../imgs/wind_turbine.gif', 'rb').read()
        self.button = widgets.Button(description='Start (Id: %d)' % self.turbine_id)
        self.button.on_click(self.__on_button_clicked__)
        
        self.img = widgets.Image(value=self.stopped_img,width=150, height=170)
        self.status_label = widgets.Label(
            layout={'width': "150px"}, value='Model loaded'
        )

        self.vibration_status = widgets.Valid(value=False, description='Vibration')
        self.voltage_status = widgets.Valid(value=False, description='Voltage')
        self.rotation_status = widgets.Valid(value=False, description='Rotation')        
        
        self.noise_buttons = [    
            widgets.Button(description='Volt', layout={'width': '50px'}),
            widgets.Button(description='Rot', layout={'width': '50px'}),
            widgets.Button(description='Vib', layout={'width': '50px'})
        ]
        for i in self.noise_buttons: i.on_click(self.__on_noise_button_clicked)
        self.anomaly_status = widgets.VBox([
            self.vibration_status, self.voltage_status, self.rotation_status,
            widgets.Label("Inject noise"),
            widgets.HBox(self.noise_buttons)            
        ], layout={'visibility': 'hidden'})          

        # subscribe to messages from inference app
        self.turbine_update_label_topic = f'wind-turbine/{turbine_id}/label/update'
        self.mqtt_client.subscribe(self.turbine_update_label_topic, mqtt.QoS.AT_LEAST_ONCE, handler=self.__callback_update_label__)

        self.turbine_anomalies_topic = f'wind-turbine/{turbine_id}/anomalies'
        self.mqtt_client.subscribe(self.turbine_anomalies_topic, mqtt.QoS.AT_LEAST_ONCE, handler=self.__callback_update_anomalies__)

        # self.mqtt_client.subscribe_to_topics(self.turbine_id,
        #                                     self.callback_update_label,
        #                                     self.callback_update_anomalies)
        
        self.feature_ids = np.array([8,9,10,7,  22, 5, 6]) # qX,qy,qz,qw  ,wind_seed_rps, rps, voltage  
        self.feature_names = np.array(['qx', 'qy', 'qz', 'qw', 'wind speed rps', 'rps', 'voltage'])
        self.data_buffer = {}
        self.raw_data_topic = 'wind-turbine/'+str(self.turbine_id)+'/raw-data'
        self.max_buffer_size = 500
        for j in range(self.max_buffer_size):
                self.__read_next_turbine_sample__()


    def __on_noise_button_clicked(self, btn):
        # change color when enabled/disabled
        btn.style.button_color = 'lightgreen' if btn.style.button_color is None else None
        
    def __on_button_clicked__(self, _):
        """ Deals with the event of Starting / Stopping the Turbine"""
        if self.halted:
            return
        
        if not self.running:
            self.running = True
            self.button.description = 'Stop (Id: %d)' % self.turbine_id            
            self.img.value = self.running_img
            self.publishing_raw_data = threading.Thread(target=self.__publish_raw_data_forver__)
            self.publishing_raw_data.start()
        else:
            self.running = False
            self.button.description = 'Start (Id: %d)' % self.turbine_id
            self.img.value = self.stopped_img
            self.publishing_raw_data.join()
        
        self.update_label(self.status_label.value)

        
    def __publish_raw_data_forver__(self):
        while self.running:
            self.__read_next_turbine_sample__()
            self.mqtt_client.publish(self.raw_data_topic, self.data_buffer)
            time.sleep(0.02)
    
    
    def __read_next_turbine_sample__(self):
        """
        Read next sample from the turbine
        """
        self.__prep_turbine_sample__(self.read_next_sample())   


        
    def __prep_turbine_sample__(self, data):
        """
        Inject noise if enabled, while preparing turbine sample data
        """
        vib_noise,rot_noise,vol_noise = self.__is_noise_enabled_for_any_type__()   
        if vib_noise: data[self.feature_ids[0:4]] = np.random.rand(4) * 100 # out of the radians range
        if rot_noise: data[self.feature_ids[5]] = np.random.rand(1) * 100 # out of the normalized wind range
        if vol_noise: data[self.feature_ids[6]] = int(np.random.rand(1)[0] * 10000) # out of the normalized voltage range

        self.data_buffer = { self.feature_names[i]:data[self.feature_ids[i]] for i in range(self.feature_ids.size)}
#         self.data_buffer[turbine_id] = x
        

    def read_next_sample(self):        
        """ next step in this simulation """
        if self.raw_data_idx >= len(self.raw_data): self.raw_data_idx = 0
        sample = self.raw_data[self.raw_data_idx]
        self.raw_data_idx += 1
        return sample
    
    
    def __is_noise_enabled_for_any_type__(self):
        """
        Check if noise is enabled for each turbine
        """
        return [self.is_noise_enabled('Vib'),
                self.is_noise_enabled('Rot'),
                self.is_noise_enabled('Vol')]

    
    def is_noise_enabled(self, typ):
        """ Returns the status of the 'inject noise' buttons (pressed or not)"""
        assert(typ == 'Vol' or typ == 'Rot' or typ == 'Vib')
        idx = 0
        if typ == 'Vol': idx = 0
        elif typ == 'Rot': idx = 1
        elif typ == 'Vib': idx = 2
        return self.noise_buttons[idx].style.button_color == 'lightgreen'

    
    def is_running(self):
        return self.running
    

            
        
    def detected_anomalies(self, values, anomalies ):
        """ Updates the status of the 'inject noise' buttons (pressed or not)"""
        self.vibration_status.value = not anomalies[0:3].any()
        self.voltage_status.value = not anomalies[3:5].any()
        self.rotation_status.value = not anomalies[5]



    def halt(self):
        """ Halts the turnine and disable it. After calling this method you can't use it anymore."""
        self.running = False
        self.button.description = 'Halted'
        self.img.value = self.stopped_img
        self.anomaly_status.layout.visibility='hidden'
        self.halted = True  

        
    def show(self):
        """ Return a IPython Widget that will render the turbine inside the notebook """
        return widgets.VBox([
            self.img, self.button, self.status_label, self.anomaly_status
        ])
    

    def __callback_update_label__(self, topic, payload, dup, qos, retain, **kwargs):
        """
        Callback when turbine receives new data to be updated on turbine label from the inference app
        """
        print("got the label update")
        json_response = json.loads(payload)
        model_label_status = json_response['model_label_status']
        self.update_label(model_label_status)


    def update_label(self, value):
        print("status label value is: ", value)
        self.status_label.value = value
        if self.is_running():
            self.anomaly_status.layout.visibility='visible'
        else:
            self.anomaly_status.layout.visibility='hidden'


    def __callback_update_anomalies__(self, topic, payload, dup, qos, retain, **kwargs):
        """
        Callback when turbine receives anomaly data from the inference app
        """
        print("got anomaly from inferece app")
        json_response = json.loads(payload)
        self.detected_anomalies(np.array(json_response['values']), np.array(json_response['anomalies']))

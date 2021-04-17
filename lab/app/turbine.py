# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import ipywidgets as widgets
import random
import time

class WindTurbine(object):
    """ Represents virtually and graphically a wind turbine
        It uses the raw data collected from a Wind Turbine in a circular buffer
        to simulate the real turbine sensors.
    """
    def __init__(self, turbine_id=0, raw_data=None):
        if raw_data is None or len(raw_data) == 0:
            raise Exception("You need to pass an array with at least one row for raw data")

        self.turbine_id = turbine_id # id of the turbine
        self.raw_data = raw_data # buffer with the raw sensors data
        self.raw_data_idx = random.randint(0, len(raw_data)-1)
        
        self.running = False # running status
                
        self.halted = False # if True you can't use this turbine anymore. create a new one.

        # components of the UI
        self.stopped_img = open('imgs/wind_turbine.png', 'rb').read()
        self.running_img = open('imgs/wind_turbine.gif', 'rb').read()
        self.button = widgets.Button(description='Start (Id: %d)' % self.turbine_id)
        self.button.on_click(self.__on_button_clicked)
        
        self.img = widgets.Image(value=self.stopped_img,width=150, height=170)
        self.status_label = widgets.Label(
            layout={'width': "150px"}, value=''
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
            
    def __on_noise_button_clicked(self, btn):
        # change color when enabled/disabled
        btn.style.button_color = 'lightgreen' if btn.style.button_color is None else None
        
    def __on_button_clicked(self, _):
        """ Deals with the event of Starting / Stopping the Turbine"""
        if self.halted:
            return
        
        if not self.running:
            self.running = True
            self.button.description = 'Stop (Id: %d)' % self.turbine_id            
            self.img.value = self.running_img            
        else:
            self.running = False
            self.button.description = 'Start (Id: %d)' % self.turbine_id
            self.img.value = self.stopped_img
        self.update_label(self.status_label.value)

    def is_running(self):
        return self.running
    
    def update_label(self, value):
        self.status_label.value = value
        if self.is_running() and value.startswith('Model Loaded'):
            self.anomaly_status.layout.visibility='visible'
        else:
            self.anomaly_status.layout.visibility='hidden'
            
        
    def detected_anomalies(self, values, anomalies ):
        """ Updates the status of the 'inject noise' buttons (pressed or not)"""
        self.vibration_status.value = not anomalies[0:3].any()
        self.voltage_status.value = not anomalies[3:5].any()
        self.rotation_status.value = not anomalies[5]

    def is_noise_enabled(self, typ):
        """ Returns the status of the 'inject noise' buttons (pressed or not)"""
        assert(typ == 'Vol' or typ == 'Rot' or typ == 'Vib')
        idx = 0
        if typ == 'Vol': idx = 0
        elif typ == 'Rot': idx = 1
        elif typ == 'Vib': idx = 2
        return self.noise_buttons[idx].style.button_color == 'lightgreen'

    def halt(self):
        """ Halts the turnine and disable it. After calling this method you can't use it anymore."""
        self.running = False
        self.button.description = 'Halted'
        self.img.value = self.stopped_img
        self.anomaly_status.layout.visibility='hidden'
        self.halted = True                    

    def read_next_sample(self):        
        """ next step in this simulation """
        if self.raw_data_idx >= len(self.raw_data): self.raw_data_idx = 0
        sample = self.raw_data[self.raw_data_idx]
        self.raw_data_idx += 1
        return sample
        
    def show(self):
        """ Return a IPython Widget that will render the turbine inside the notebook """
        return widgets.VBox([
            self.img, self.button, self.status_label, self.anomaly_status
        ])

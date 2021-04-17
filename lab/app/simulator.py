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

class WindTurbineFarmSimulator(object):
    def __init__(self, n_turbines=5):
        self.n_turbines = n_turbines
        
        # read the raw data. This data was captured from real sensors installed in the mini Wind Turbine
        self.raw_data = pd.read_csv('data/dataset_wind.csv.gz', compression="gzip", sep=',', low_memory=False).values
        
        # now create the virtual wind turbines
        self.turbines = [WindTurbine(i, self.raw_data) for i in range(n_turbines)]
        self.data_buffer = [[] for i in range(n_turbines)]
        self.running = False
        self.agents = None
        self.halted = False     
        
        self.feature_ids = np.array([8,9,10,7,  22, 5, 6]) # qX,qy,qz,qw  ,wind_seed_rps, rps, voltage        
        self.feature_names = np.array(['qx', 'qy', 'qz', 'qw', 'wind speed rps', 'rps', 'voltage'])
        self.colors = np.array([ 'r', 'g', 'y', 'b', 'r', 'g', 'b'])
        
        self.max_buffer_size = 500
        for idx in range(n_turbines):
            for j in range(self.max_buffer_size):
                self.__read_next_turbine_sample__(idx)

        self.dashboard = widgets.Textarea(value='\n' * self.n_turbines, disabled=True,
            layout={'border': '1px solid black', 'width': '850px', 'height': '90px'})

    def __del__(self):
        self.halt()
        
    def __launch_agent__(self, agent_id):
        """
        Launches Linux processes for each Edge Agent. 
        They will run in background and listen to a unix socket
        """
        # remove channel
        subprocess.Popen(["rm", "-f", "/tmp/agent%d" % agent_id])
        # launch main process
        cmd = "./agent/bin/sagemaker_edge_agent_binary -c agent/conf/config_edge_device_%d.json -a /tmp/agent%d" % (agent_id, agent_id)
        logs = open("agent/logs/agent%d.log" % agent_id, "+w")
        # we need to return the process in order to terminate it later
        return subprocess.Popen(cmd.split(' '), stdout=logs)

    def __prep_turbine_sample__(self, turbine_id, data):
        vib_noise,rot_noise,vol_noise = self.is_noise_enabled(turbine_id)
        #np.array([8,9,10,7,  22, 5, 6]) # qX,qy,qz,qw  ,wind_seed_rps, rps, voltage        
        if vib_noise: data[self.feature_ids[0:4]] = np.random.rand(4) * 100 # out of the radians range
        if rot_noise: data[self.feature_ids[5]] = np.random.rand(1) * 100 # out of the normalized wind range
        if vol_noise: data[self.feature_ids[6]] = int(np.random.rand(1)[0] * 10000) # out of the normalized voltage range

        self.data_buffer[turbine_id].append(data)
        if len(self.data_buffer[turbine_id]) > self.max_buffer_size:
            del self.data_buffer[turbine_id][0]
    
    def __read_next_turbine_sample__(self, turbine_id):
        self.__prep_turbine_sample__(turbine_id, self.turbines[turbine_id].read_next_sample() )
        
    def is_turbine_running(self, turbine_id):
        return self.turbines[turbine_id].is_running()
    
    def show(self):        
        return widgets.VBox([
            widgets.HBox([t.show() for t in self.turbines]),
            self.dashboard
        ])

    def update_dashboard(self, turbine_id, data):
        if not self.turbines[turbine_id].is_running(): return
        lines = self.dashboard.value.split('\n')        
        features = np.mean(data[-50:,self.feature_ids], axis=0)
        tokens = ["%s: %0.3f" % (self.feature_names[i], features[i]) for i in range(len(features))]
        lines[turbine_id] = ' '.join(["Turbine: %d" % turbine_id] + tokens)        
        self.dashboard.value = '\n'.join(lines)

    def start(self):
        """
        Run the main application by creating the Edge Agents, loading the model and
        kicking-off the anomaly detector program
        """
        if not self.running and not self.halted:
            self.running = True
            logging.info("Launching Edge Manager Agents...")
            self.agents = [self.__launch_agent__(i) for i in range(self.n_turbines)]
            logging.info("Agents launched! (waiting 5 secs)")
        time.sleep(5) # give some time for the agents to launch
    
    def halt(self):
        if self.running:
            self.running = False
            self.halted = True
            # halt all the turbines
            for i in self.turbines: i.halt()            
            # kill the agents
            for i in self.agents: 
                #os.kill(i.pid, signal.SIGINT)
                i.kill()
                i.wait()

    def get_num_turbines(self):
        return self.n_turbines
    
    def get_raw_data(self, turbine_id):        
        assert(turbine_id >= 0 and turbine_id < len(self.data_buffer))
        self.__read_next_turbine_sample__(turbine_id)
        return self.data_buffer[turbine_id]
    
    def detected_anomalies(self, turbine_id, values, anomalies):
        assert(turbine_id >= 0 and turbine_id < len(self.data_buffer))
        self.turbines[turbine_id].detected_anomalies(values, anomalies)

    def update_label(self, turbine_id, value ):
        self.turbines[turbine_id].update_label(value)

    def is_noise_enabled(self, turbine_id):
        return [self.turbines[turbine_id].is_noise_enabled('Vib'),
                self.turbines[turbine_id].is_noise_enabled('Rot'),
                self.turbines[turbine_id].is_noise_enabled('Vol')]


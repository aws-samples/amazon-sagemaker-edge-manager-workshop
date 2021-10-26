# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import threading
import numpy as np
import logging
import time
import json
import sys
import typing
from edgeagentclient import EdgeAgentClient
import util
import messaging_client as msg_client
import os


class WindTurbine(object):
    """ 
    This is the application class. It is responsible for:
        - Creating virtual edge device (as thread)
        - Launch Edge Agent in virtual device
        - Load the Anomaly detection for the Wind Turbine in the Edge Agent
        - Launch the Virtual Wind Turbines
        - Launch a Edge Agent Client that integrates the Wind Turbine with the Edge Device
    """
    # extra args model_path, model_name, model_version
    def __init__(self, turbine_id, agent_socket, model_path, model_name):
        if turbine_id is None:
            raise Exception("You need to pass the turbine id as argument")
        
        self.running = False # running status

        self.msg_client = msg_client.MessagingClient(turbine_id)
        self.msg_client.subscribe_to_data(self.__data_handler__)
#         self.msg_client.subscribe_to_status(self.__callback_is_turbine_running__)

        ## launch edge agent client
        self.edge_agent = EdgeAgentClient(agent_socket)
        self.model_meta = {
            "model_name" : model_name,
            "model_path" : model_path
        }

        self.acc_buffer = []   
        self.dashboard_buffer = []
        self.model_loaded = False

        self.resp = self.edge_agent.load_model(model_name, model_path)
        if self.resp is None: 
            logging.error('It was not possible to load the model. Is the agent running?')
            sys.exit(1)
        self.model_loaded = True 
        model_loaded= {"model_label_status" : "Model Loaded"}
        self.msg_client.publish_model_status(model_loaded)
        

        # we need to load the statistics computed in the data prep notebook
        # these statistics will be used to compute normalize the input
        file_path = os.path.dirname(__file__)
        logging.info(f"Reading stats from {file_path}")
        self.raw_std = np.load(os.path.join(file_path, '../statistics/raw_std.npy'))
        self.mean = np.load(os.path.join(file_path, '../statistics/mean.npy'))
        self.std = np.load(os.path.join(file_path, '../statistics/std.npy'))
        # then we load the thresholds computed in the training notebook
        # for more info, take a look on the Notebook #2
        self.thresholds = np.load(os.path.join(file_path, '../statistics/thresholds.npy'))
        
        # configurations to format the time based data for the anomaly detection model
        # If you change these parameters you need to retrain your model with the new parameters        
        self.INTERVAL = 5 # seconds
        self.TIME_STEPS = 20 * self.INTERVAL # 50ms -> seg: 50ms * 20
        self.STEP = 10
        
        # these are the features used in this application
        self.feature_ids = [8,9,10,7,  22, 5, 6] # qX,qy,qz,qw  ,wind_seed_rps, rps, voltage        
        self.n_features = 6 # roll, pitch, yaw, wind_speed, rotor_speed, voltage
        
        
        # minimal buffer length for denoising. We need to accumulate some sample before denoising
        self.min_num_samples = 500

#     def __callback_is_turbine_running__(self, topic_name, payload):
#         """
#         gets the running status from turbines and update their status in turbine holder
#         """
#         json_response = json.loads(payload)
#         turbine_status = json_response['running']
        
#         if turbine_status is True or turbine_status is False:
#             self.running = turbine_status
#         else:
#             print("Invalid turbine status recieved from simulator, ignoring it, current status is: ", self.running)
            
        
       

    def __del__(self):
        """Destructor"""
        self.halt()

    def __data_handler__(self, topic, payload):
        """
        Subscription handler for the data topic
        Turbine sends the data on an MQTT topic which is handled by this function
        that accumulates the data in a temporary buffer before processing. 
        Implements a rolling-window logic.

        {
            "qx": 0,
            "qy": -0.01,
            "qz": 0.99,
            "qw": -0.1,
            "wind speed rps": 2.06,
            "rps": 2.19,
            "voltage": 70
        }
        """

        json_response = json.loads(payload)
        raw_data = np.array(list(json_response.values()))
        self.acc_buffer.append(raw_data)
        self.dashboard_buffer.append(raw_data.tolist())
        if len(self.acc_buffer) >= self.min_num_samples:
            logging.info("Got enough samples - detecting anomalies")
            new_buf = self.acc_buffer.copy()
            self.acc_buffer = []
            
            # update the dashboard of simulator
            self.msg_client.publish_data(self.dashboard_buffer)
            self.dashboard_buffer = []
            
            self.__detect_anomalies__(new_buf)

            
    def __detect_anomalies__(self, buffer):     
        """
        Process the data received from the turbine and reports the 
        anomalies detected via MQTT
        """
        
        start_time = time.time()

        # create a copy & prep the data
        data = self.__data_prep__(np.array(buffer))
            
        if not self.edge_agent.is_model_loaded(self.model_meta['model_name']):
            model_label_data = {"model_label_status" : "Model not loaded"}
            self.msg_client.publish_model_status(model_label_data)
            return

        x = self.__preprocess_data__(data)
        # run the model                    
        p = self.edge_agent.predict(self.model_meta['model_name'], x)
        
        if p is not None:
            values, anomalies = self.__calculate_anomalies__(x, p)
            anomaly_result = {"values" : values.tolist(), "anomalies" : anomalies.tolist()} 
            self.msg_client.publish_anomalies(message=anomaly_result)   
        else:
            logging.info(f"No anomalies detected")

        elapsed_time = time.time() - start_time
        time.sleep(0.5-elapsed_time)


    def __data_prep__(self, buffer):
        """
        This method is called for each reading.
        Here we do some data prep and accumulate the data in the buffer
        for denoising
        """

        # We already receive the the necessary data so, remove the feature_ids mappings and use the indexes directly
        new_buffer = []
        for data in buffer:
            roll,pitch,yaw = util.euler_from_quaternion(
                data[0],data[1],
                data[2],data[3]
            )
            row = [roll,pitch,yaw, data[4],data[5], data[6]]
            new_buffer.append(row)
        return np.array(new_buffer)        



    def __preprocess_data__(self, data):
        # denoise
        data = np.array([util.wavelet_denoise(data[:,i], 'db6', self.raw_std[i]) for i in range(self.n_features)])
        data = data.transpose((1,0))
        # normalize                     
        data -= self.mean
        data /= self.std
        data = data[-(self.TIME_STEPS+self.STEP):]
        # create the dataset and reshape it
        x = util.create_dataset(data, self.TIME_STEPS, self.STEP)                    
        x = np.transpose(x, (0, 2, 1)).reshape(x.shape[0], self.n_features, 10, 10)
        return x
    
    
    def __calculate_anomalies__(self, x, p):
        a = x.reshape(x.shape[0], self.n_features, 100).transpose((0,2,1))
        b = p.reshape(p.shape[0], self.n_features, 100).transpose((0,2,1))
        # check the anomalies
        pred_mae_loss = np.mean(np.abs(b - a), axis=1).transpose((1,0))
        values = np.mean(pred_mae_loss, axis=1)
        anomalies = (values > self.thresholds)
        return values, anomalies   
     

    def start(self):
        """
        Run the main application by creating the Edge Agent, loading the model and
        kicking-off a thread to keep the program running since the processing is event based
        """
        if not self.running:
            self.running = True
        logging.info("Waiting for data...")
        while self.running:
            time.sleep(0.1)
            
                # finally start the anomaly detection loop

    def halt(self):
        """
        Destroys the application
        """
        self.running = False
            
            
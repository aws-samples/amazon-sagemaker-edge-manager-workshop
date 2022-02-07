# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from awscrt import io, mqtt, auth, http
from awsiot import mqtt_connection_builder
import time as t
import json
import boto3
import numpy as np
import logging

"""
This class represents an MQTT client which connects to IoT MQTT 
and provides utility methods for publish and subscribe to topics.
"""

class Client():
    def __init__(self, client_id):
        self.client_id = client_id
        
    def connect(self):
        """
        Method to connect to IoT MQTT via IAM mode.
        It uses the current exection role to setup the connection.
        """
        event_loop_group = io.EventLoopGroup()
        host_resolver = io.DefaultHostResolver(event_loop_group)
        client_bootstrap = io.ClientBootstrap(event_loop_group, host_resolver)
        credentials_provider = auth.AwsCredentialsProvider.new_default_chain(
            client_bootstrap)

        iot_client = boto3.client('iot')
        endpoint = iot_client.describe_endpoint(
            endpointType='iot:Data-ATS')['endpointAddress']
        region = endpoint.split(".")[2]

        mqtt_connection = mqtt_connection_builder.websockets_with_default_aws_signing(
            endpoint=endpoint,
            client_bootstrap=client_bootstrap,
            client_id=self.client_id,
            region=region,
            credentials_provider=credentials_provider,
            clean_session=False)

        logging.info("Connecting to {} with client ID '{}'...".format(endpoint, self.client_id))
        # Make the connect() call
        connect_future = mqtt_connection.connect()
        # Future.result() waits until a result is available
        connect_future.result()
        logging.info("Connected!")
        self.mqtt_connection = mqtt_connection
        return True


    

    def publish(self, topic, data):
        """
        Publish message to IoT MQTT client.
        """
        json_payload = json.dumps(data)
        self.mqtt_connection.publish(
            topic, json_payload, mqtt.QoS.AT_LEAST_ONCE)

        
    # def subscribe_to_topics(self, turbine_id, callback_update_label, callback_update_anomalies):
    #     """
    #     Used by WindTurbine class to subscribe to topics coming from inference app
    #     """
    #     try:
    #         turbine_update_label_topic = f'wind-turbine/{turbine_id}/label/update'
    #         self.subscribe(turbine_update_label_topic, mqtt.QoS.AT_LEAST_ONCE, handler=callback_update_label)

    #         turbine_anomalies_topic = f'wind-turbine/{turbine_id}/anomalies'
    #         self.subscribe(turbine_anomalies_topic, mqtt.QoS.AT_LEAST_ONCE, handler=callback_update_anomalies)
    #     except Exception as ex:
    #         raise ex


    def subscribe(self, topic, qos, handler):
        """
        Subscribe to topics coming from inference app
        """
        try:
            subscribe_future, packet_id = self.mqtt_connection.subscribe(topic=topic, qos=qos, callback=handler)
            subscribe_future.result()
            logging.info("Subscribed to {}".format(topic))   
        except Exception as ex:
            raise ex

        

    def disconnect(self):
        """
        Stop the connection to IoT MQTT
        """
        disconnect_future = self.mqtt_connection.disconnect()
        disconnect_future.result()
        logging.info("Disconnected from MQTT")
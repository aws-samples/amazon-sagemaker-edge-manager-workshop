import json
import ggv2_client as ggv2
from awsiot.greengrasscoreipc.model import QOS
import logging
"""
client for subscribing and processing messages from the windturbine devices
"""


class MessagingClient():
    def __init__(self, turbine_id):
        self.turbine_id = turbine_id
        self.turbine_update_dashboard_topic = f'wind-turbine/{turbine_id}/dashboard/update'
        self.turbine_update_label_topic = f'wind-turbine/{turbine_id}/label/update'
        self.turbine_anomalies_topic = f'wind-turbine/{turbine_id}/anomalies'
        self.turbine_raw_data_topic = f'wind-turbine/{turbine_id}/raw-data'


    def subscribe_to_data(self, handler):
        """
        Subscribes to topics publishing turbine data for a single turbine
        """
        ggv2.subscribe(topic=self.turbine_raw_data_topic, 
                        qos=QOS.AT_LEAST_ONCE, 
                        handler=handler)

    def publish_anomalies(self, message):
        json_message = json.dumps(message)
        ggv2.publish(topic=self.turbine_anomalies_topic,
                message=bytes(json_message, 'utf-8'),
                qos=QOS.AT_LEAST_ONCE)

    def publish_model_status(self, message):
        json_message = json.dumps(message)
        ggv2.publish(topic=self.turbine_update_label_topic,
                message=bytes(json_message, 'utf-8'),
                qos=QOS.AT_LEAST_ONCE)

    def publish_data(self, message):
        json_message = json.dumps(message)
        ggv2.publish(topic=self.turbine_update_dashboard_topic,
                message=bytes(json_message, 'utf-8'),
                qos=QOS.AT_LEAST_ONCE)



    

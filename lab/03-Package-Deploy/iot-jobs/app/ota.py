# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import ssl
import paho.mqtt.client as mqtt
import logging
import json
import os
import io
import time
import requests
import boto3
import tarfile
import glob
import threading

class OTAModelUpdate(object):
    def __init__(self, device_id, device_name, mqtt_host, mqtt_port, update_callback, model_path='agent/model'):
        assert(update_callback != None)
        logging.basicConfig(filename='agent/logs/ota.log', encoding='utf-8', level=logging.DEBUG)
        
        self.device_id = device_id
        self.device_name = device_name
        self.model_path = model_path
        self.update_callback = update_callback
        conf_file_name = 'agent/conf/config_%s.json' % device_name.replace('-', '_')
        if not os.path.isfile(conf_file_name):
            raise Exception("Agent config file not found! %s" % conf_file_name)
        params=json.loads(open(conf_file_name, 'r').read())
        self.params = params
        self.mqttc = mqtt.Client()
        self.mqttc.tls_set(
            params['sagemaker_edge_provider_aws_ca_cert_file'],
            certfile=params['sagemaker_edge_provider_aws_cert_file'], 
            keyfile=params['sagemaker_edge_provider_aws_cert_pk_file'], 
            cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLSv1_2, ciphers=None
        )
        self.mqttc.enable_logger(logger=logging)
        self.mqttc.on_message = self.__on_message__
        self.mqttc.on_connect = self.__on_connect__
        self.mqttc.on_disconnect = self.__on_disconnect__
        self.connected = False
        
        self.model_meta = {'model_name': None}
        for f in glob.glob(os.path.join(model_path, str(device_id), '*', '*', 'compiled.*')):
            tokens = f.split(os.path.sep)
            assert(len(tokens) > 3)
            name = tokens[-3]
            version = float(tokens[-2])
            if self.model_meta['model_name'] != name or self.model_meta['model_version'] < version:
                self.model_meta['model_name'] = name
                self.model_meta['model_version'] = version
        logging.info("Model meta", self.model_meta)

        if self.model_meta['model_name'] is not None:
            self.update_callback(self.device_id, self.model_meta['model_name'], self.model_meta['model_version'])
        
        self.processing_lock = threading.Lock()
        self.processed_jobs = []
        
        # start the mqtt client
        self.mqttc.connect(mqtt_host, mqtt_port, 45)
        self.mqttc.loop_start()
        
    def publish(self, topic, payload=None):
        self.mqttc.publish(topic, payload)
                
    def get_client(self, service_name):
        access_key_id,secret_access_key,session_token = self.__get_aws_credentials__(
            self.params['sagemaker_edge_provider_aws_iot_cred_endpoint'],
            self.device_name,
            self.params['sagemaker_edge_provider_aws_cert_file'],
            self.params['sagemaker_edge_provider_aws_cert_pk_file'] )
        
        return boto3.client(
            service_name, self.params['sagemaker_edge_core_region'],
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            aws_session_token=session_token
        )

    def model_update_check(self):
        if self.connected:
            self.mqttc.publish('$aws/things/%s/jobs/get' % self.device_name)

    def __get_aws_credentials__(self, cred_endpoint, thing_name, cert_file, key_file):
        resp = requests.get(
            cred_endpoint,
            headers={'x-amzn-iot-thingname': thing_name}, 
            cert=(cert_file, key_file),
        )
        if not resp:
            logging.error("Something went wrong when I tried to get new credentials: %s" % resp)
            raise Exception('Error while getting the IoT credentials: ', resp)
        credentials = resp.json()
        return (credentials['credentials']['accessKeyId'],
            credentials['credentials']['secretAccessKey'],
            credentials['credentials']['sessionToken'])
    
    def __del__(self):
        logging.info("Deleting this object")
        self.mqttc.loop_stop()
        self.mqttc.disconnect()
        
    def __update_job_status__(self, job_id, status, details):
        payload = json.dumps({
            "status": status,
            "statusDetails": {"info": details },
            "includeJobExecutionState": False,
            "includeJobDocument": False,
            "stepTimeoutInMinutes": 2,
        })
        logging.info("Updating IoT job status: %s" % details)
        self.publish('$aws/things/%s/jobs/%s/update' % ( self.device_name, job_id), payload)
        
    def __on_message__(self, client, userdata, message):
        logging.debug("New message. Topic: %s; Message: %s;" % (message.topic, message.payload))
        
        if message.topic.endswith('notify'):
            self.model_update_check()

        elif message.topic.endswith('accepted'):
            resp = json.loads(message.payload)            
            logging.debug(resp)
            if resp.get('queuedJobs') is not None: # request to list jobs
                # get the description of each queued job
                for j in resp['queuedJobs']:
                    ## get the job description
                    self.publish('$aws/things/%s/jobs/%s/get' % ( self.device_name, j['jobId'] ) )
                    break
            elif resp.get('inProgressJobs') is not None: # request to list jobs
                # get the description of each queued job
                for j in resp['inProgressJobs']:
                    ## get the job description
                    self.publish('$aws/things/%s/jobs/%s/get' % ( self.device_name, j['jobId'] ) )
                    break
            elif resp.get('execution') is not None: # request to get job description            
                # check if this is a job description message
                job_meta = resp.get('execution')
            
                # we have the job metadata, let's process it
                self.__update_job_status__(job_meta['jobId'], 'IN_PROGRESS', 'Trying to get/load the model')
                self.__process_job__(job_meta['jobId'], job_meta['jobDocument'])
            else:
                logging.debug('Other message: ', resp)
                
    def __process_job__(self, job_id, msg):
        self.processing_lock.acquire()
        if job_id in self.processed_jobs:
            self.processing_lock.release()
            return
        self.processed_jobs.append(job_id)
        try:
            if msg.get('type') == 'new_model':                
                model_version = float(msg['model_version'])
                model_name = msg['model_name']

                if self.model_meta.get('model_name') is not None:
                    if self.model_meta['model_name'] != model_name:
                        msg = 'New model name doesnt match the current name: %s' % model_name
                        logging.info(msg)
                        self.__update_job_status__(job_id, 'FAILED', msg)
                        self.processing_lock.release()
                        return
                        
                    if self.model_meta['model_version'] >= model_version:
                        msg = "New model version is not newer than the current one. Curr: %f; New: %f;" % (self.model_meta['model_version'], model_version)
                        logging.info(msg)
                        self.__update_job_status__(job_id, 'FAILED', msg)
                        self.processing_lock.release()
                        return

                logging.info("Downloading new model package")
                s3_client = self.get_client('s3')
                
                package = io.BytesIO(s3_client.get_object(
                    Bucket=msg['model_package_bucket'], 
                    Key=msg['model_package_key'])['Body'].read()
                )
                logging.info("Unpacking model package")
                with tarfile.open(fileobj=package) as p:
                    p.extractall(os.path.join(self.model_path, str(self.device_id), msg['model_name'], msg['model_version']))
                    self.model_meta['model_name'] = msg['model_name']
                    self.model_meta['model_version'] = model_version
                                
                self.__update_job_status__(job_id, 'SUCCEEDED', 'Model deployed')
                self.update_callback(self.device_id, self.model_meta['model_name'], self.model_meta['model_version'])                
            else:
                logging.info("Model '%s' version '%f' is the current one or it is obsolete" % (self.model_metadata['model_name'], self.model_metadata['model_version']))
        except Exception as e:
            self.__update_job_status__(job_id, 'FAILED', str(e))            
            logging.error(e)

        self.processing_lock.release()
        
    def __on_connect__(self, client, userdata, flags, rc):
        self.connected = True
        logging.info("Connected!")
        #self.mqttc.subscribe('$aws/things/%s/jobs/notify-next' % self.device_name)
        self.mqttc.subscribe('$aws/things/%s/jobs/notify' % self.device_name)
        self.mqttc.subscribe('$aws/things/%s/jobs/accepted' % self.device_name)
        self.mqttc.subscribe('$aws/things/%s/jobs/rejected' % self.device_name)
        time.sleep(1)
        self.model_update_check()

    def __on_disconnect__(self, client, userdata, flags):
        self.connected = False
        logging.info("Disconnected!")

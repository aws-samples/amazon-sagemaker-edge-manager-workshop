# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import os
import stat
import boto3
import json
import argparse
import logging
import tarfile
import requests
import io
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

iot_client = boto3.client('iot')
sm_client = boto3.client('sagemaker')
s3_client = boto3.client('s3')

def setup_agent(agent_id, args, thing_group_name, thing_group_arn):
    device_prefix = 'edge-device-wind-turbine-%011d' # it needs to have 36 chars
    policy_name='WindTurbineFarmPolicy-%s' % args.sagemaker_project_id
    base="agent/certificates/iot/edge_device_%d_%s.pem"
    fleet_name = 'wind-turbine-farm-%s' % args.sagemaker_project_id
    thing_arn_template = thing_group_arn.replace('thinggroup', 'thing').replace(thing_group_name, '%s')
    cred_host = iot_client.describe_endpoint(endpointType='iot:CredentialProvider')['endpointAddress']
    policy_alias = 'SageMakerEdge-%s' % fleet_name

    # register the device in the fleet    
    # the device name needs to have 36 chars
    dev_name = "edge-device-wind-turbine-%011d" % agent_id
    dev = [{'DeviceName': dev_name, 'IotThingName': "edge-device-%d" % agent_id}]    
    try:        
        sm_client.describe_device(DeviceFleetName=fleet_name, DeviceName=dev_name)
        logger.info("Device was already registered on SageMaker Edge Manager")
    except ClientError as e:
        if e.response['Error']['Code'] != 'ValidationException': raise e
        logger.info("Registering a new device %s on fleet %s" % (dev_name, fleet_name))
        sm_client.register_devices(DeviceFleetName=fleet_name, Devices=dev)
        iot_client.add_thing_to_thing_group(
            thingGroupName=thing_group_name,
            thingGroupArn=thing_group_arn,
            thingName='edge-device-%d' % agent_id,
            thingArn=thing_arn_template % ('edge-device-%d' % agent_id)
        )        
    
    # if you reach this point you need to create new certificates
    # generate the certificates    
    cert=base % (agent_id, 'cert')
    key=base % (agent_id, 'pub')
    pub=base % (agent_id, 'key')
    
    cert_meta=iot_client.create_keys_and_certificate(setAsActive=True)
    cert_arn = cert_meta['certificateArn']
    with open(cert, 'w') as c: c.write(cert_meta['certificatePem'])
    with open(key,  'w') as c: c.write(cert_meta['keyPair']['PrivateKey'])
    with open(pub,  'w') as c: c.write(cert_meta['keyPair']['PublicKey'])
        
    # attach the certificates to the policy and to the thing
    iot_client.attach_policy(policyName=policy_name, target=cert_arn)
    iot_client.attach_thing_principal(thingName='edge-device-%d' % agent_id, principal=cert_arn)        
    
    logger.info("Finally, let's create the agent config file")
    agent_params = {
        "sagemaker_edge_core_device_name": device_prefix % agent_id,
        "sagemaker_edge_core_device_fleet_name": fleet_name,
        "sagemaker_edge_core_capture_data_buffer_size": 30,
        "sagemaker_edge_core_capture_data_batch_size": 10,
        "sagemaker_edge_core_capture_data_push_period_seconds": 4,
        "sagemaker_edge_core_folder_prefix": "wind_turbine_data",
        "sagemaker_edge_core_region": args.aws_region,
        "sagemaker_edge_core_root_certs_path": "./agent/certificates/root",
        "sagemaker_edge_provider_aws_ca_cert_file":"./agent/certificates/iot/AmazonRootCA1.pem",
        "sagemaker_edge_provider_aws_cert_file":"./%s" % cert,
        "sagemaker_edge_provider_aws_cert_pk_file":"./%s" % key,
        "sagemaker_edge_provider_aws_iot_cred_endpoint": "https://%s/role-aliases/%s/credentials" % (cred_host,policy_alias),
        "sagemaker_edge_provider_provider": "Aws",
        "sagemaker_edge_provider_s3_bucket_name": args.artifact_bucket,
        "sagemaker_edge_core_capture_data_destination": "Cloud"
    }
    with open('agent/conf/config_edge_device_%d.json' % agent_id, 'w') as conf:
        conf.write(json.dumps(agent_params, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", type=str, default=os.environ.get("LOGLEVEL", "INFO").upper())
    parser.add_argument("--sagemaker-project-id", type=str, required=True)
    parser.add_argument("--sagemaker-project-name", type=str, required=True)    
    parser.add_argument("--artifact-bucket", type=str, required=True)
    parser.add_argument("--aws-region", type=str, required=True)
    parser.add_argument("--num-agents", type=int, default=5)
    
    args, _ = parser.parse_known_args()
    
    # Configure logging to output the line number and message
    log_format = "%(levelname)s: [%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(format=log_format, level=args.log_level)
            
    # create a new thing group
    thing_group_name='WindTurbineFarm-%s' % args.sagemaker_project_id
    thing_group_arn = None
    agent_pkg_bucket = 'sagemaker-edge-release-store-us-west-2-linux-x64'
    agent_config_package_prefix = 'wind_turbine_agent/config.tgz'

    # Create model package group if necessary
    try:
        # check if the model package group exists
        resp = sm_client.describe_model_package_group(ModelPackageGroupName=args.sagemaker_project_name)
        model_package_group_arn = resp['ModelPackageGroupArn']
    except ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException':
            # it doesn't exist, lets create a new one
            resp = sm_client.create_model_package_group(
                ModelPackageGroupName=args.sagemaker_project_name,
                ModelPackageGroupDescription="Wind Turbine Anomaly model group",
                Tags=[
                    {'Key': 'sagemaker:project-name', 'Value': args.sagemaker_project_name},
                    {'Key': 'sagemaker:project-id', 'Value': args.sagemaker_project_id},
                ]
            )
            model_package_group_arn = resp['ModelPackageGroupArn']
        else:
            raise e
            
    try:
        s3_client.download_file(Bucket=args.artifact_bucket, Key=agent_config_package_prefix, Filename='/tmp/dump')
        logger.info('The agent configuration package was already built! Skipping...')
        quit()
    except ClientError as e:
        pass
         
    try:
        thing_group_arn = iot_client.describe_thing_group(thingGroupName=thing_group_name)['thingGroupArn']
        logger.info("Thing group found")
    except iot_client.exceptions.ResourceNotFoundException as e:
        logger.info("Creating a new thing group")
        thing_group_arn = iot_client.create_thing_group(thingGroupName=thing_group_name)['thingGroupArn']

    logger.info("Creating the directory structure for the agent")
    # create a structure for the agent files
    os.makedirs('agent/certificates/root', exist_ok=True)
    os.makedirs('agent/certificates/iot', exist_ok=True)
    os.makedirs('agent/logs', exist_ok=True)
    os.makedirs('agent/model', exist_ok=True)
    os.makedirs('agent/conf', exist_ok=True)
    
    # then get some root certificates
    resp = requests.get('https://www.amazontrust.com/repository/AmazonRootCA1.pem')
    with open('agent/certificates/iot/AmazonRootCA1.pem', 'w') as c:
        c.write(resp.content.decode('utf-8'))
    
    # this certificate validates the edge manage package
    s3_client.download_file(
        Bucket=agent_pkg_bucket, 
        Key='Certificates/%s/%s.pem' % (args.aws_region, args.aws_region), 
        Filename='agent/certificates/root/%s.pem' % args.aws_region
    )
    # adjust the permissions of the files
    os.chmod('agent/certificates/iot/AmazonRootCA1.pem', stat.S_IRUSR|stat.S_IRGRP)
    os.chmod('agent/certificates/root/%s.pem' % args.aws_region, stat.S_IRUSR|stat.S_IRGRP)
    
    logger.info("Processing the agents...")
    for agent_id in range(args.num_agents): setup_agent(agent_id, args, thing_group_name, thing_group_arn )
    
    logger.info("Creating the final package...")
    with io.BytesIO() as f:
        with tarfile.open(fileobj=f, mode='w:gz') as tar:
            tar.add('agent', recursive=True)
        f.seek(0)
        logger.info("Uploading to S3")
        s3_client.upload_fileobj(f, Bucket=args.artifact_bucket, Key=agent_config_package_prefix) 
    logger.info("Done!")


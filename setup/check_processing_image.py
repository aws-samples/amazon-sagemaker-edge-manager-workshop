# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import boto3
import argparse

ecr_client = boto3.client('ecr')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-name', type=str, default='sagemaker-wind-turbine-processing')
    args, _ = parser.parse_known_args()

    repo_name=args.repo_name
    image_found = False
    try:
        ecr_client.describe_images(repositoryName=repo_name)    
        for i in ecr_client.list_images(repositoryName=repo_name)['imageIds']:
            if i['imageTag'] == 'latest':
                image_found = True
                break
        print('IMAGE_FOUND') if image_found else print('IMAGE_NOT_FOUND')
    except ecr_client.exceptions.RepositoryNotFoundException as e:
        print('REPO_NOT_FOUND')

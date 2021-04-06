# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import sys
import subprocess
# we need a special package for cleaning our data, lets pip install it first
subprocess.check_call([sys.executable, "-m", "pip", "install", "pywavelets==1.1.1"])

import argparse
import os
import warnings

import pandas as pd
import numpy as np
import pywt
import math
import glob
from datetime import datetime

def create_dataset(X, time_steps=1, step=1):
    """
    Encode the timeseries dataset into a
    multidimentional tensor in the format: num_features x step x step.
    It uses a time window approach to slide on 'step' right in the timeseries.    
    """
    Xs = []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
    return np.array(Xs)

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z # in radians

def wavelet_denoise(data, wavelet, noise_sigma):
    '''Filter accelerometer data using wavelet denoising

    Modification of F. Blanco-Silva's code at: https://goo.gl/gOQwy5
    '''
    
    wavelet = pywt.Wavelet(wavelet)
    levels  = min(5, (np.floor(np.log2(data.shape[0]))).astype(int))
    
    # Francisco's code used wavedec2 for image data
    wavelet_coeffs = pywt.wavedec(data, wavelet, level=levels)
    threshold = noise_sigma*np.sqrt(2*np.log2(data.size))

    new_wavelet_coeffs = map(lambda x: pywt.threshold(x, threshold, mode='soft'), wavelet_coeffs)

    return pywt.waverec(list(new_wavelet_coeffs), wavelet)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=int, default=5)
    parser.add_argument('--time-steps', type=int, default=20)
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--num-dataset-splits', type=int, default=25)
    args, _ = parser.parse_known_args()

    print('Received arguments {}'.format(args))

    INTERVAL = args.interval # seconds
    TIME_STEPS = args.time_steps * INTERVAL # Xms -> seg: Xms * Y
    STEP = args.step

    # selected the features
    features = ['roll', 'pitch', 'yaw', 'wind_speed_rps', 'rps', 'voltage']    
    input_data_base_path = '/opt/ml/processing/input'
    stats_output_base_path = '/opt/ml/processing/statistics'
    dataset_output_base_path = '/opt/ml/processing/train'

    # load the data file
    parser = lambda date: datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.%f+00:00')
    dfs = []
    for f in glob.glob(os.path.join(input_data_base_path, '*.gz')):
        dfs.append(pd.read_csv(f, compression='gzip', sep=',', low_memory=False, parse_dates=[ 'eventTime'], date_parser=parser))
    df = pd.concat(dfs)
    
    print('now converting quat to euler...')
    roll,pitch,yaw = [], [], []
    for idx, row in df.iterrows():
        r,p,y = euler_from_quaternion(row['qx'], row['qy'], row['qz'], row['qw'])
        roll.append(r)
        pitch.append(p)
        yaw.append(y)
    df['roll'] = roll
    df['pitch'] = pitch
    df['yaw'] = yaw

    df = df[features] # keep only the selected features
    
    # get the std for denoising
    raw_std = df.std()
    # denoise
    for f in features:
        df[f] = wavelet_denoise(df[f].values, 'db6', raw_std[f])

    # normalize
    training_std = df.std()
    training_mean = df.mean()
    df = (df - training_mean) / training_std
    
    # export the dataset statistics
    np.save(os.path.join(stats_output_base_path, 'raw_std.npy'), raw_std)
    np.save(os.path.join(stats_output_base_path, 'mean.npy'), training_mean)
    np.save(os.path.join(stats_output_base_path, 'std.npy'), training_std)
    
    # format the dataset
    n_cols = len(df.columns)
    X = create_dataset(df, TIME_STEPS, STEP)
    X = np.nan_to_num(X, copy=True, nan=0.0, posinf=None, neginf=None)
    X = np.transpose(X, (0, 2, 1)).reshape(X.shape[0], n_cols, 10, 10)

    ## We need to split the array in chunks of at most 5MB    
    for i,x in enumerate(np.array_split(X, args.num_dataset_splits)):
        np.save(os.path.join(dataset_output_base_path, 'wind_turbine_%02d.npy' % i), x)
    print("Number of training samples:", len(df))

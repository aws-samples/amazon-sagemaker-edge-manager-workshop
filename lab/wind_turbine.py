# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import argparse
import glob
import numpy as np
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.optim as optim
from   torch.autograd import Variable
from   sklearn.model_selection import KFold

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_model(n_features, dropout=0):    
    return torch.nn.Sequential(
        torch.nn.Conv2d(n_features, 32, kernel_size=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Conv2d(32, 64, kernel_size=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Conv2d(64, 128, kernel_size=2, padding=2),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(128, 64, kernel_size=2, padding=2),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.ConvTranspose2d(64, 32, kernel_size=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.ConvTranspose2d(32, n_features, kernel_size=2, padding=1),
    )    

def load_data(data_dir):
    input_files = glob.glob(os.path.join(data_dir, '*.npy'))
    data = [np.load(i) for i in input_files]
    return np.vstack(data)    

def train_epoch(optimizer, criterion, epoch, model, train_dataloader, test_dataloader):
    train_loss = 0.0    
    test_loss = 0.0    
    model.train()
    for x_train, y_train in train_dataloader:
        # clearing the Gradients of the model parameters
        optimizer.zero_grad()
        # prediction for training and validation set        
        output_train = model(x_train)        
        loss_train = criterion(output_train, y_train)
                
        # computing the updated weights of all the model parameters
        # statistics
        train_loss += loss_train.item()
        loss_train.backward()
        optimizer.step()        
    model.eval()
    for x_test, y_test in test_dataloader:            
        output_test = model(x_test.float())
        loss_test = criterion(output_test, y_test)
        # statistics
        test_loss += loss_test.item()                
        
    return train_loss, test_loss

def train(args):
    best_of_the_best = (0,-1)
    best_loss = 10000000
    num_epochs = args.num_epochs
    batch_size = args.batch_size    
    
    X = load_data(args.train)
    criterion = nn.MSELoss()    
    kf = KFold(n_splits=args.k_fold_splits, shuffle=True)
    num_features = X.shape[1]
    
    for i, indexes in enumerate(kf.split(X)):
        # skip other Ks if fixed was informed
        if args.k_index_only >= 0 and args.k_index_only != i: continue

        train_index, test_index = indexes
        print("Test dataset proportion: %.02f%%" % (len(test_index)/len(train_index) * 100))
        X_train, X_test = X[train_index], X[test_index]
        X_train = torch.from_numpy(X_train).float().to(device)
        X_test = torch.from_numpy(X_test).float().to(device)

        train_dataset = torch.utils.data.TensorDataset(X_train, X_train)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        test_dataset = torch.utils.data.TensorDataset(X_test, X_test)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

        model = create_model(num_features, args.dropout_rate)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        # Instantiate model
        # Training loop
        for epoch in range(num_epochs):
            start_time = time.time()
            train_loss, test_loss = train_epoch( optimizer, criterion, epoch, model, train_dataloader, test_dataloader)
            elapsed_time = (time.time() - start_time)
            print("k=%d; epoch=%d; train_loss=%.3f; test_loss=%.3f; elapsed_time=%.3fs" % (i, epoch, train_loss, test_loss, elapsed_time))
            if test_loss < best_loss:                
                torch.save(model.state_dict(), os.path.join(args.output_data_dir,'model_state.pth'))
                best_loss = test_loss
                if best_loss < best_of_the_best[0]:
                    best_of_the_best = (best_loss, i)
    print("\nBest model: best_mse=%f;" % best_loss)
    model = create_model(num_features, args.dropout_rate)
    model.load_state_dict( torch.load(os.path.join(args.output_data_dir, "model_state.pth")) )
    os.mkdir(os.path.join(args.model_dir,'code'))
    shutil.copyfile(__file__, os.path.join(args.model_dir, 'code/inference.py'))
    torch.save(model, os.path.join(args.model_dir, "model.pth"))


def model_fn(model_dir):
    model = torch.load(os.path.join(model_dir, "model.pth"))
    model = model.to(device)
    model.eval()
    return model

def predict_fn(input_data, model):    
    with torch.no_grad():
        return model(input_data.float().to(device))
    
if __name__ == '__main__':
    nn.DataParallel
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.    
    parser.add_argument('--k_fold_splits', type=int, default=6)
    parser.add_argument('--k_index_only', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)    
    parser.add_argument('--learning_rate', type=float, default=0.003)
    parser.add_argument('--dropout_rate', type=float, default=0.0)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()
    train(args)

#! /home/rennerph/.conda/envs/framatome/bin/python

import numpy as np
import os 
import csv

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import optuna
print(optuna.__version__)
from FaceDetection import FaceDetection
from generator import CustomDataset
import argparse

from Transformer import ViViT

class Postprocessing():
    def __init__(self): 
        self.data_list = os.listdir("data/Train64")

        np.random.shuffle(self.data_list)
        train_size = int(len(self.data_list) * 0.9)
        self.train_data = self.data_list[:train_size]
        self.validation_data = self.data_list[train_size:]

        print(f"Length train data {len(self.train_data)}")
        print(f"Length valid data {len(self.validation_data)}")
    
        self.first_call = True
        self.window_length = 2
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")

        if self.device.type == "mps":
            self.local_flag = True
        else:
            self.local_flag = False


        print(f'Using device: {self.device}')
    
    
    def model_build(self):
        """
        Builds a AI model with specified parameters.
        """
        #model = ResNet(n_channels = 2, depth = depth, initial_filter = initial_filter)
        model = ViViT(64, 16, 1, 60, depth = self.depth, heads = self.heads, dim = self.dim, dim_head = self.dim_head)
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3fM' % parameters)
        return model

    def save_heart_rate_data_to_csv(self, groundtruth_hr_mean, detected_hr):
        """
        Saves ground truth heart rate mean and detected heart rate to a CSV file.

        Parameters:
        groundtruth_hr_mean (float): Mean of ground truth heart rate data.
        detected_hr (float): Detected heart rate value to save.

        Returns:
        None
        """
        with open('heartrate_data.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([groundtruth_hr_mean, detected_hr])    


    def processing(self):
        """
        Performs video processing and heart rate detection for each participant using the specified model parameters.

        Parameters:
        depth (int): Depth parameter for model construction.

        Returns:
        float: The mean loss value across all participants and epochs.
        """
        if self.local_flag == True:
            import tensorflow as tf

        train_dataset = CustomDataset(self.train_data, fft_filter=self.fft_filter)
        val_dataset = CustomDataset(self.validation_data, fft_filter=self.fft_filter)

        batch_size = 8

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
 
        model = self.model_build()
        model = model.to(self.device)

        loss_epoch = []
        num_epochs = 20
        for epoch in range(num_epochs):
            print("##########################################")
            print(f"EPOCH   :   {epoch} of {num_epochs}")
            print("##########################################")
            
            optimizer = optim.Adam(model.parameters(), lr=5e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2)
            
            train_loss = []
            train_mae = []

            if self.local_flag == True:
                progress_bar = tf.keras.utils.Progbar(target = len(train_loader))

            step = 0
            for data in train_loader:
                step = step + 1
                x_train, y_train = data
                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)
                optimizer.zero_grad()

                heart_rate_detected = model(x_train)
                heart_rate_detected = torch.squeeze(heart_rate_detected)
        
                #heart_rate_detected = self.eularian_functions.find_heart_rate(video=video_ai_processed, freq_min = 0, freq_max=5, fps=fps)

                loss =  torch.nn.functional.mse_loss(heart_rate_detected, y_train)
                mae_loss = torch.nn.functional.l1_loss(heart_rate_detected, y_train)
                loss.backward()
                optimizer.step()

                heart_rate_detected = heart_rate_detected.cpu().detach().numpy().astype(np.float32)
                heart_rate_detected = torch.unsqueeze(torch.tensor(heart_rate_detected).to(torch.float32), dim = 0)

                loss = float(loss.cpu().detach().numpy())
                mae_loss = float(mae_loss.cpu().detach().numpy())
                train_loss.append(loss)
                train_mae.append(mae_loss)

                if self.local_flag == True:
                    progress_bar.update(step, values=[("loss mse", loss),("loss mae", mae_loss)])

            with torch.no_grad():
                val_loss = []
                val_mae = []

            if self.local_flag == True:
                progress_bar = tf.keras.utils.Progbar(target = len(val_loader))

                step = 0
                for data in val_loader:
                    step = step + 1
                    x_val, y_val = data
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)
                    heart_rate_detected = model(x_val)
                    heart_rate_detected = torch.squeeze(heart_rate_detected)
                    loss =  torch.nn.functional.mse_loss(heart_rate_detected, y_val)
                    mae_loss = torch.nn.functional.l1_loss(heart_rate_detected, y_val)
                    loss = float(loss.cpu().detach().numpy())
                    mae_loss = float(mae_loss.cpu().detach().numpy())
                    val_loss.append(loss)
                    val_mae.append(mae_loss)

                    scheduler.step(loss)

                    if self.local_flag == True:
                        progress_bar.update(step, values=[("loss mse", loss),("loss mae", mae_loss)])

            print("______________________________________________________________________________________________________________")
            print(F"Epoch:   {epoch} of {num_epochs}")
            print(F"Train loss mse:   {np.mean(train_loss)}      Validation Loss mse:    {np.mean(val_loss)}")
            print(F"Train loss mae:   {np.mean(train_mae)}       Validation Loss mae:    {np.mean(val_mae)}")
            print("______________________________________________________________________________________________________________")

        return np.mean(val_loss)

    def objective(self, trial):
        """
        Objective function for Optuna optimization.

        Parameters:
        trial (optuna.Trial): A Trial object that stores parameters and intermediate results.

        Returns:
        float: The computed loss value to minimize.
        """
        self.depth = trial.suggest_categorical('depth', [6,8,12,16])
        self.heads = trial.suggest_categorical('heads', [6,16,32,64])
        self.dim = trial.suggest_categorical('dim', [192, 256, 512])
        self.dim_head = trial.suggest_categorical('dim_head', [64])
        self.fft_filter = trial.suggest_categorical('fft_filter', [True])

        print(f"depth:   {self.depth}")
        print(f"heads:   {self.heads}")
        print(f"dim:   {self.dim}")
        print(f"dim_head:   {self.dim_head}")
        print(f"fft_filter:   {self.fft_filter}")

        loss = self.processing()
        print(f"Iteration {trial.number}: error = {loss}")
        return loss


if __name__ == "__main__":
    postprocessing = Postprocessing()
    study = optuna.create_study(direction='minimize')
    study.optimize(postprocessing.objective, n_trials=50)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print(f"  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
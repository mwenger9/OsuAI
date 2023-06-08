import torch
from torch import nn
from LSTMModel import LSTMModel
from DatasetLoader import DataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import re 

from tqdm import tqdm

"""
Boucle d'entraiment du modèle.
Récapitulatif :

Modele : LSTM

INPUT_FEATURES  = ['x', 'y','visible', 'is_slider', 'is_spinner']
OUTPUT_FEATURES = ['x', 'y','K1',"K2"]

Size of one batch : 32 (Meaning beatmaps are processed 32 at a time)

Loss function : MSE

Optimizer : Adam



"""



import numpy as np

if __name__ == "__main__":
    NUM_EPOCHS = 200
    VALIDATION_SPLIT = 0.1
    EVAL_EVERY = 5  # Evaluate every 5 epochs
    
    DATASET_PATH = "D:\\osu!rdr_dataset"
    beatmap_folder = f"{DATASET_PATH}\\beatmaps"
    replay_folder = f"{DATASET_PATH}\\replays\\osr"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_input_features = 5
    num_output_features = 2

    print("Creating the model")
    model = LSTMModel(num_input_features=num_input_features,num_output_features=num_output_features)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(f"Using device {device}")
    model = model.to(device)

    model.train()
    torch.cuda.empty_cache()

    extract_number_in_name = r".*_(\d{1,3}).pt"
    tensor_directory = "D:\\osu!rdr_dataset\\preprocessed_tensors_32"

    input_files = sorted((f for f in os.listdir(tensor_directory) if f.startswith("input_tensor_")))
    target_files = sorted((f for f in os.listdir(tensor_directory) if f.startswith("target_tensor_")))
    zipped_tensors = list(zip(input_files, target_files))

    # Split data into training and validation sets
    train_tensors, val_tensors = train_test_split(zipped_tensors, test_size=VALIDATION_SPLIT, random_state=42)

    print("Start of the training loop")
    for epoch in range(1,NUM_EPOCHS+1):
        print(f"Starting epoch {epoch}")
        for i, (input_filenames, target_filenames) in tqdm(enumerate(train_tensors), total=len(train_tensors), desc=f"Epoch {epoch}"):
            inputs = torch.load(f'{tensor_directory}\\{input_filenames}').float().to(device)
            targets = torch.load(f'{tensor_directory}\\{target_filenames}').float()
            targets = targets[:, :, :-2]  # keeps all rows and all elements in the sequence, but only the first N-2 features / Remove k1,k2
            targets = targets.to(device)


            outputs = model(inputs)

            mask = torch.sum(targets, dim=2) != 0  
            outputs = outputs[mask]
            targets = targets[mask]

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}, Loss: {loss.item()}')

        if epoch % EVAL_EVERY == 0:
            model.eval()
            val_loss = []
            with torch.no_grad():
                for input_filenames, target_filenames in val_tensors:
                    inputs = torch.load(f'{tensor_directory}\\{input_filenames}').float().to(device)
                    targets = torch.load(f'{tensor_directory}\\{target_filenames}').float()
                    targets = targets[:, :, :-2]  # keeps all rows and all elements in the sequence, but only the first N-2 features
                    targets = targets.to(device)

                    outputs = model(inputs)
                    mask = torch.sum(targets, dim=2) != 0
                    outputs = outputs[mask]
                    targets = targets[mask]

                    loss = criterion(outputs, targets)
                    val_loss.append(loss.item())

            print(f'Validation Loss after {epoch} epochs: {np.mean(val_loss)}')
            model.train()

        torch.save(model.state_dict(), f'C:\\Users\\tehre\\Desktop\\INSA\\S6\\IA_Jeux\\weights\\model_weights_{epoch}.pth')

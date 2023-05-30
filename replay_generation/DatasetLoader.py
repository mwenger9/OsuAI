from keras.utils import Sequence,pad_sequences
import numpy as np
import os
import pandas as pd
from parse_osu_map_file import parse_beatmap_file,parse_beatmap_file_and_chunk
from parse_osr_file import parse_osr_file,parse_osr_file_and_chunk,parse_osr_file_and_chunk_aligned
import torch


"""
Le but de cette classe est de créer des batch de données pour le LSTM.

L'idée est de permettre au LSTM de travailler sur un paquet de données à chaque fois (ici 8 beatmaps par 8)
étant donnée qu'on peut pas juste charger tout le dataset en mémoire à cause de sa taille. 


"""

class DataGenerator(Sequence):
    def __init__(self, df, beatmap_folder, replay_folder, batch_size=32):
        'Initialization'
        self.df = df
        self.beatmap_folder = beatmap_folder
        self.replay_folder = replay_folder
        self.batch_size = batch_size
        self.save_folder = "D:\\osu!rdr_dataset\\preprocessed_datas"
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        start = index*self.batch_size
        end = (index+1)*self.batch_size
        rows = self.df.iloc[start:end]
        X, Y = self.__data_generation(rows)
        return torch.from_numpy(np.array(X)), torch.from_numpy(np.array(Y))


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.df))

    def __data_generation(self, rows):
        X = []
        Y = []
        for beatmap_hash,replay_hash in zip(rows["beatmapHash"],rows["replayHash"]):

            beatmap_file = os.path.join(self.beatmap_folder, f'{beatmap_hash}.osu')
            replay_file = os.path.join(self.replay_folder, f'{replay_hash}.osr')
            try:
                inputs = parse_beatmap_file_and_chunk(beatmap_file) 
                targets = parse_osr_file_and_chunk_aligned(replay_file,inputs)
                # print("global shapes :")
                # print(inputs.shape,targets.shape)  

                # Assuming each input and target is a list of chunks,
                # append each chunk as a separate input-target pair
                for input_chunk, target_chunk in zip(inputs, targets):
                    # print("chunk shapes : ")
                    # print(input_chunk.shape,target_chunk.shape)
                    X.append(input_chunk)
                    Y.append(target_chunk)

            except:
                print(f"error while parsing {beatmap_file} or {replay_file}, it will not be added to the preprocessed datas")

        return X, Y
    

    def preprocess_and_save_data(self, rows, path="D:\\osu!rdr_dataset\\preprocessed_datas"):
        for beatmap_hash, replay_hash in zip(rows["beatmapHash"],rows["replayHash"]):

            beatmap_file = os.path.join(self.beatmap_folder, f'{beatmap_hash}.osu')
            replay_file = os.path.join(self.replay_folder, f'{replay_hash}.osr')
            try:
                inputs = parse_beatmap_file_and_chunk(beatmap_file)  
                targets = parse_osr_file_and_chunk_aligned(replay_file)  

                # Assuming each input and target is a list of chunks,
                # save each chunk as a separate tensor
                for i, (input_chunk, target_chunk) in enumerate(zip(inputs, targets)):
                    input_tensor = torch.from_numpy(input_chunk)
                    target_tensor = torch.from_numpy(target_chunk)
                    # Save tensors to disk
                    torch.save(input_tensor, os.path.join(path, f'{beatmap_hash}_{replay_hash}_{i}_input.pt'))
                    torch.save(target_tensor, os.path.join(path, f'{beatmap_hash}_{replay_hash}_{i}_target.pt'))

            except:
                print(f"error while parsing {beatmap_file} or {replay_file}, it will not be added to the preprocessed datas")


    



### UTILISÉ POUR TESTER LE FONCTIONNEMENT DE LA CLASSE

# if __name__ == '__main__':
#     from sklearn.model_selection import train_test_split

#     DATASET_PATH = "D:\\osu!rdr_dataset"
#     beatmap_folder = f"{DATASET_PATH}\\beatmaps\\"
#     replay_folder = f"{DATASET_PATH}\\replays\\osr\\"
#     df = pd.read_csv(f'{DATASET_PATH}\\final_cleaned_index.csv')

#     batch_size = 32
#     # Assuming df is your full dataset
#     train_df, validation_df = train_test_split(df, test_size=0.2)

#     # Create data generators for training and validation data
#     training_data_generator = DataGenerator(train_df, beatmap_folder, replay_folder, batch_size=batch_size)
#     validation_data_generator = DataGenerator(validation_df, beatmap_folder, replay_folder, batch_size=batch_size)
    
#     # Fetch one batch of data
#     train_X, train_y = training_data_generator.__getitem__(0)
#     val_X, val_y = validation_data_generator.__getitem__(0)

#     # Print shapes
#     print("Training data:")
#     print("X:", train_X.shape)
#     print("y:", train_y.shape)

#     print("\nValidation data:")
#     print("X:", val_X.shape)
#     print("y:", val_y.shape)

#     train_X, train_y = training_data_generator.__getitem__(1)
#     val_X, val_y = validation_data_generator.__getitem__(1)


#     print("Training data:")
#     print("X:", train_X.shape)
#     print("y:", train_y.shape)

#     print("\nValidation data:")
#     print("X:", val_X.shape)
#     print("y:", val_y.shape)


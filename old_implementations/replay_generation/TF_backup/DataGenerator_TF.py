from keras.utils import Sequence,pad_sequences
import numpy as np
import os
import pandas as pd
from parse_osu_map_file import parse_beatmap_file
from parse_osr_file import parse_osr_file
import torch

class DataGenerator(Sequence):
    def __init__(self, df, beatmap_folder, replay_folder, batch_size=32):
        'Initialization'
        self.df = df
        self.beatmap_folder = beatmap_folder
        self.replay_folder = replay_folder
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        start = index*self.batch_size
        end = (index+1)*self.batch_size
        print(start,end)
        rows = self.df.iloc[start:end]
        X, Y = self.__data_generation(rows)
        return np.array(X), np.array(Y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.df))

    def __data_generation(self, rows):
        X = []
        Y = []
        for _,row in rows.iterrows():
            beatmap_hash = row["beatmapHash"]
            replay_hash = row["replayHash"]

            beatmap_file = os.path.join(self.beatmap_folder, f'{beatmap_hash}.osu')
            replay_file = os.path.join(self.replay_folder, f'{replay_hash}.osr')
            try:
                inputs = parse_beatmap_file(beatmap_file)  
                targets = parse_osr_file(replay_file)
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

    


# if __name__ == "__main__":

#     DATASET_PATH = "D:\\osu!rdr_dataset"
#     beatmap_folder = f"{DATASET_PATH}\\beatmaps\\"
#     replay_folder = f"{DATASET_PATH}\\replays\\osr\\"
#     df = pd.read_csv(f'{DATASET_PATH}\\final_cleaned_index.csv')


    

#     batch_size = 32


#     validation_data_generator = DataGenerator(df, beatmap_folder, replay_folder, batch_size=batch_size)

#     print(validation_data_generator)


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split

    DATASET_PATH = "D:\\osu!rdr_dataset"
    beatmap_folder = f"{DATASET_PATH}\\beatmaps\\"
    replay_folder = f"{DATASET_PATH}\\replays\\osr\\"
    df = pd.read_csv(f'{DATASET_PATH}\\final_cleaned_index.csv')

    batch_size = 8
    # Assuming df is your full dataset
    train_df, validation_df = train_test_split(df, test_size=0.2)

    # Create data generators for training and validation data
    training_data_generator = DataGenerator(train_df, beatmap_folder, replay_folder, batch_size=batch_size)
    validation_data_generator = DataGenerator(validation_df, beatmap_folder, replay_folder, batch_size=batch_size)
    
    # Fetch one batch of data
    train_X, train_y = training_data_generator.__getitem__(0)
    val_X, val_y = validation_data_generator.__getitem__(0)

    # Print shapes
    print("Training data:")
    print("X:", train_X.shape)
    print("y:", train_y.shape)

    print("\nValidation data:")
    print("X:", val_X.shape)
    print("y:", val_y.shape)

    train_X, train_y = training_data_generator.__getitem__(1)
    val_X, val_y = validation_data_generator.__getitem__(1)


    print("Training data:")
    print("X:", train_X.shape)
    print("y:", train_y.shape)

    print("\nValidation data:")
    print("X:", val_X.shape)
    print("y:", val_y.shape)

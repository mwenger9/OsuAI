from keras.utils import Sequence
import numpy as np
import pandas as pd
import dataset
import torch


"""
Classe utilisée pour générer les batchs de données pour le model.

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
        print(f"creating chunk from {start} to {end}")
        rows = self.df.iloc[start:end]


        rows = dataset.load(rows)
        try :
            input_data = dataset.input_data(rows)
            target_data = dataset.target_data(rows)
        except : 
            return None,None
        X = np.reshape(input_data.values, (-1, dataset.BATCH_LENGTH, len(dataset.INPUT_FEATURES)))
        Y = np.reshape(target_data.values, (-1, dataset.BATCH_LENGTH, len(dataset.OUTPUT_FEATURES)))
        return torch.from_numpy(X), torch.from_numpy(Y)


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.df))




if __name__ == '__main__':
    from sklearn.model_selection import train_test_split

    DATASET_PATH = "D:\\osu!rdr_dataset"
    beatmap_folder = f"{DATASET_PATH}\\beatmaps"
    replay_folder = f"{DATASET_PATH}\\replays\\osr"
    df = pd.read_csv(f'{DATASET_PATH}\\beatmap_replay_index.csv')




    batch_size = 100
    train_df, validation_df = train_test_split(df, test_size=0.2)

    training_data_generator = DataGenerator(train_df, beatmap_folder, replay_folder, batch_size=batch_size)
    validation_data_generator = DataGenerator(validation_df, beatmap_folder, replay_folder, batch_size=batch_size)
    
    # Fetch one batch of data
    train_X, train_y = training_data_generator.__getitem__(0)
    val_X, val_y = validation_data_generator.__getitem__(0)

    # Print shapes
    print("Training data:")
    print("X:", train_X.shape)
    print("Y:", train_y.shape)

    print("\nValidation data:")
    print("X:", val_X.shape)
    print("Y:", val_y.shape)

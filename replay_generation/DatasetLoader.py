from keras.utils import Sequence,pad_sequences
import numpy as np
import os
import pandas as pd
from parse_osu_map_file import parse_beatmap_file,parse_beatmap_file_and_chunk
from parse_osr_file import parse_osr_file,parse_osr_file_and_chunk,parse_osr_file_and_chunk_aligned
import torch

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

    # def __getitem__(self, index):
    #     # Select sample
    #     beatmap_hash = self.df['beatmapHash'].iloc[index]
    #     replay_hash = self.df['replayHash'].iloc[index]
    #     chunk_number = self.df['chunkNumber'].iloc[index]  # Assuming you've added this column

    #     # Load data and get label
    #     input_path = os.path.join(self.save_folder, f'{beatmap_hash}_{replay_hash}_{chunk_number}_input.pt')
    #     target_path = os.path.join(self.save_folder, f'{beatmap_hash}_{replay_hash}_{chunk_number}_target.pt')

    #     inputs = torch.load(input_path)
    #     targets = torch.load(target_path)

    #     return inputs, targets


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

    batch_size = 32
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





# import torch
# from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
# from torch.utils.data import Dataset
# import numpy as np
# import os
# from parse_osu_map_file import parse_beatmap_file
# from parse_osr_file import parse_osr_file
# import pandas as pd

# class OsuDataset(Dataset):

#     DATASET_PATH = "D:\\osu!rdr_dataset"
#     BEATMAP_FOLDER = f"{DATASET_PATH}\\beatmaps\\"
#     REPLAY_FOLDER = f"{DATASET_PATH}\\replays\\osr\\"
#     INDEX_PATH = f"{DATASET_PATH}\\final_cleaned_index.csv"

#     MAX_LENGTH = 75

#     def __init__(self, csv_file=INDEX_PATH, beatmap_folder=BEATMAP_FOLDER, replay_folder=REPLAY_FOLDER):
#         self.df = pd.read_csv(csv_file)
#         self.beatmap_folder = beatmap_folder
#         self.replay_folder = replay_folder

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):

#         beatmap_hash = self.df.iloc[idx]['beatmapHash']
#         replay_hash = self.df.iloc[idx]['replayHash']

#         beatmap_file = os.path.join(self.beatmap_folder, f'{beatmap_hash}.osu')
#         replay_file = os.path.join(self.replay_folder, f'{replay_hash}.osr')

#         inputs = parse_beatmap_file(beatmap_file)  
#         targets = parse_osr_file(replay_file)  

#         inputs = [self.check_and_convert(i, is_target=False) for i in inputs]
#         targets = [self.check_and_convert(t, is_target=True) for t in targets]

#         return inputs, targets
    

#     def _pad_sequence(sequence, max_length=MAX_LENGTH):
#         padded_sequence = sequence
#         if len(sequence) < max_length:
#             padded_sequence = torch.nn.functional.pad(
#                 sequence, (0, 0, 0, max_length - len(sequence)))
#         else:
#             padded_sequence = sequence[:max_length]
#         return padded_sequence
    

    
#     def dict_to_tensor(self, dict_obj):

#         if 'Type' in dict_obj:  # it's a beatmap
#             time = torch.tensor([dict_obj['Time']], dtype=torch.float32)
#             type_ = torch.tensor(dict_obj['Type'], dtype=torch.float32)
#             x = torch.tensor([dict_obj['x']], dtype=torch.float32)
#             y = torch.tensor([dict_obj['y']], dtype=torch.float32)
#             slider_type = torch.tensor(dict_obj['Slider Type'], dtype=torch.float32)
#             slider_points = torch.tensor(dict_obj['Slider Points'], dtype=torch.float32)
#             if slider_points.shape[0] < self.MAX_LENGTH:
#                 # Pad slider_points tensor with zeroes up to MAX_LENGTH
#                 padding = torch.zeros((self.MAX_LENGTH - slider_points.shape[0], 2), dtype=torch.float32)
#                 slider_points = torch.cat([slider_points, padding])

#             elif slider_points.shape[0] > self.MAX_LENGTH:
        
#                 slider_points = slider_points[:self.MAX_LENGTH, :]
                
#             slider_length = torch.tensor([dict_obj['Slider Length']], dtype=torch.float32)

#             # Concatenate all tensors into one
#             tensor = torch.cat([time, type_, x,y, slider_type, slider_points.flatten(), slider_length])

#         else:  # it's a replay
#             time_delta = torch.tensor([dict_obj['time_delta']], dtype=torch.float32)
#             x = torch.tensor([dict_obj['x']], dtype=torch.float32)
#             y = torch.tensor([dict_obj['y']], dtype=torch.float32)
#             keys = torch.tensor(dict_obj['keys'], dtype=torch.float32)
            
#             tensor = torch.cat([time_delta, x, y, keys])

#         return tensor
    

#     def tensor_to_dict(self, tensor):
#         dict_obj = {
#             'time_delta': tensor[0].item(),
#             'x': tensor[1].item(),
#             'y': tensor[2].item(),
#             'keys': tensor[3:].tolist()
#         }
#         return dict_obj
    
#     def collate_fn(self, batch):
#         inputs = []
#         targets = []
        
#         for item in batch:
#             try:
#                 input_, target_ = item
                
#                 input_ = torch.stack(input_)
#                 target_ = torch.stack(target_)

#                 inputs.append(input_)
#                 targets.append(target_)
#             except Exception as e:
#                 print(f"An exception occurred while processing an item: {e}")
#                 continue  # skip this item
        
#         # Pad sequences for inputs and targets
#         inputs_padded = pad_sequence(inputs, batch_first=True)
#         targets_padded = pad_sequence(targets, batch_first=True)

#         return inputs_padded, targets_padded
        

#     def check_and_convert(self, dict_obj, is_target=False):
#         required_keys_inputs = {"Time": (int, float, np.float16), "Type": list, "x": np.float16,"y":np.float16, 
#                                 "Slider Type": list, "Slider Points": (list,np.ndarray), "Slider Length": (int, float, np.float16)}
#         required_keys_targets = {"time_delta": (int, float, np.float16), "x": (int, float, np.float16), 
#                                 "y": (int, float, np.float16), "keys": list}

#         if not is_target:  # it's a beatmap
#             for key, type_ in required_keys_inputs.items():
#                 if key not in dict_obj:
#                     print(f"Key {key} missing in input dictionary: {dict_obj}")
#                     return None
#                 if not isinstance(dict_obj[key], type_):
#                     print(f"Value of key {key} in input dictionary is not of correct type: {type(dict_obj[key])}")
#                     return None
#         else:  # it's a replay
#             for key, type_ in required_keys_targets.items():
#                 if key not in dict_obj:
#                     print(f"Key {key} missing in target dictionary: {dict_obj}")
#                     return None
#                 if not isinstance(dict_obj[key], type_):
#                     print(f"Value of key {key} in target dictionary is not of correct type: {type(dict_obj[key])}")
#                     return None

#         return self.dict_to_tensor(dict_obj)

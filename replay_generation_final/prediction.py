import numpy as np
import torch
from LSTMModel import LSTMModel
import dataset
from beatmap import load as beatmap_load
import pandas as pd
from osrparse import Replay,Mod,ReplayEventOsu,Key,GameMode
import re 
import requests
import datetime
"""
Script permettant de faire une prédiction pour un fichier de beatmap donnée (beatmap_path) et son ID

"""

def reverse_normalize(coordinate_list):
    """
    Reverse normalization on the output.
    """

    return [(coordinate_list[0] + 0.5) * SCREEN_WIDTH, (coordinate_list[1] + 0.5) * SCREEN_HEIGHT]


def tensor_to_dataframe(tensor):
    # First, reshape the tensor so it's two-dimensional
    reshaped_tensor = tensor.reshape(-1, tensor.shape[-1])

    # Then, convert the tensor to a numpy array so we can use it to create a DataFrame
    numpy_array = reshaped_tensor.cpu().numpy()

    # Create a DataFrame from the numpy array
    df = pd.DataFrame(numpy_array, columns=['x', 'y'])

    return df

def predict_new_beatmap(beatmap_path, model):
    # Preprocess the beatmap


    df = pd.DataFrame({"beatmap":[beatmap_path]
                       ,"replay":['D:\\osu!rdr_dataset\\replays\\osr\\b27df1328412abf3b588bcfa6bbd8b04.osr']})
    

    loaded_datas = dataset.load(df)
    beatmap_start = loaded_datas["beatmap"].iloc[0].start_offset()
    input_data = dataset.input_data(loaded_datas)
    input_data = np.reshape(input_data.values, (-1, dataset.BATCH_LENGTH, len(dataset.INPUT_FEATURES)))
    input_data = torch.tensor(input_data).float().to(device)


    # Forward pass
    output_data = model(input_data)

    # Convert output back to numpy
    output_data = output_data.cpu().detach().numpy()

    # Reverse normalization
    #output_data = reverse_normalize(output_data)

    return beatmap_start,output_data


if __name__ == "__main__":
# Load the trained model

    BEATMAP_PATH = "D:\\Osu\\Songs\\423527 dj TAKA - quaver\\dj TAKA - quaver (Monstrata) [Crescendo].osu"
    BEATMAP_ID = 915210
    API_KEY = "0375dc7e7d29a37993082d62a4776c90bec512c7"



    beatmap_start = 0
    # Model parameters
    num_input_features = 5
    num_output_features = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Normalization used : 

    SCREEN_WIDTH = 512
    SCREEN_HEIGHT = 384

    model = LSTMModel(num_input_features=num_input_features,num_output_features=num_output_features)
    model.load_state_dict(torch.load("C:\\Users\\tehre\\Desktop\\INSA\\S6\\IA_Jeux\\playfield\\model_weights_32.pth"))
    model = model.to(device)

    # Use the new beatmap's file path

    beatmap_start,output_data = predict_new_beatmap(BEATMAP_PATH, model)

    
    tensor_2d = output_data.reshape(-1, output_data.shape[-1])
    tensor_2d = list(map(reverse_normalize,tensor_2d))

    result_action_list = [[24,x,y] for x,y in tensor_2d]
    result_action_list[0][0] = beatmap_start
    # pd.DataFrame(result_action_list,columns=["time_delta","x","y","key"]).to_csv("prediction_tensor.csv",mode='w+',index=False)

    replay_event_list = [ReplayEventOsu(time_delta=time_delta,x=x,y=y,keys=Key(0)) for time_delta,x,y in result_action_list]

    # print([event for event in replay_event_list if isinstance(event,list)])
    print(beatmap_start)

    print(replay_event_list[0])

    
    
    response = requests.get(f"https://osu.ppy.sh/api/get_beatmaps?k={API_KEY}&b={BEATMAP_ID}")
    response = response.json()[0]

    beatmap_hash = response["file_md5"]

    rx = Mod(1<<7) # 1 << 7 is for relax
    replay = Replay(mode=GameMode(0),
                    game_version=20230326,
                    beatmap_hash=beatmap_hash,
                    username="LSTM",
                    count_300=0,
                    count_100=0,
                    count_50=0,
                    count_geki=0,
                    count_katu=0,
                    count_miss=0,
                    score=0,
                    max_combo=0,
                    perfect=False,
                    mods=rx,
                    timestamp=datetime.datetime.now(),
                    life_bar_graph=[],
                    replay_data=replay_event_list,
                    rng_seed=None,
                    replay_id=0,
                    replay_hash="0")
    
    replay.write_path("generated_replay.osr")



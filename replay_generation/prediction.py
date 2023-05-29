import numpy as np
import torch
from LSTMModel import LSTMModel
from parse_osu_map_file import parse_beatmap_file_and_chunk

# Model parameters
input_size = 161
hidden_size = 50
num_layers = 1
output_size = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define your normalization parameters
X_UPPER_BOUND = 600
X_LOWER_BOUND = -180
Y_LOWER_BOUND = -82
Y_UPPER_BOUND = 407
LENGTH_THRESHOLD = 360000  # Replace ... with the appropriate value

def reverse_normalize(y):
    """
    Reverse normalization on the output.
    """
    time = y[0] * LENGTH_THRESHOLD
    x = y[1] * (X_UPPER_BOUND - X_LOWER_BOUND) + X_LOWER_BOUND
    y = y[2] * (Y_UPPER_BOUND - Y_LOWER_BOUND) + Y_LOWER_BOUND
    keys = y[3:] 

    return np.array([time, x, y] + keys.tolist())

def predict_new_beatmap(beatmap_path, model):
    # Preprocess the beatmap
    input_data = parse_beatmap_file_and_chunk(beatmap_path)
    input_data = np.array(input_data)
    input_data = input_data.reshape(-1, 1, input_data.shape[-1])
    input_data = torch.tensor(input_data).float().to(device)

    # # Add an extra dimension for the batch size
    # input_data = input_data.unsqueeze(0)

    # Forward pass
    output_data = model(input_data)

    # Convert output back to numpy
    output_data = output_data.cpu().detach().numpy()

    # Reverse normalization
    #output_data = reverse_normalize(output_data)

    return output_data

# Load the trained model
model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
model.load_state_dict(torch.load('C:\\Users\\tehre\\Desktop\\INSA\\S6\\IA_Jeux\\replay_generation\\model_weights.pth'))
model = model.to(device)

# Use the new beatmap's file path
beatmap_path = "D:\\Osu\\Songs\\393995 toby fox - Quiet Water\\toby fox - Quiet Water (Intelli) [Calm].osu"

output_data = predict_new_beatmap(beatmap_path, model)
print(output_data)

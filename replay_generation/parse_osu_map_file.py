from pprint import pprint
from coordinates_utils import normalize,X_UPPER_BOUND,X_LOWER_BOUND,Y_LOWER_BOUND,Y_UPPER_BOUND
import os 
import numpy as np
from keras.utils import pad_sequences

"""
Ce script permet de parser un fichier .osu, et de récolter toutes les données sur les objets présent dans une beatmap.
Ces données sont ensuite normalisées (one-hot encoding, normalisation, padding, etc..) pour faire en sorte qu'elles soient
au bon format pour être utilisées pas le LSTM.

Ces données sont ensuite séparées en chunk par intervals de temps : 

Par exemple si l'interval de temps choisi est 1000 ms, le premier chunk contiendra les objets de la beatmap (circle,slider,..)
ayant lieu entre 0 et 1000 ms, le deuxième entre 1000ms et 2000ms, etc..


Format des données :

Un objet (circle,slider) correspond aux données suivantes : 

161 features (Beatmap file)
Time (Normalized)
x (Normalized)
y (Normalized)
Type (One hot encoding)
Slider type (One hot encoding)
Slider length (One hot encoding)
Slider points (capped to 75 -> 75 * (x,y) coordinates = 150)


"""



LENGTH_THRESHOLD = 360000 
SLIDER_LENGTH_THRESHOLD = 3000
MAX_SLIDER_POINTS = 75
CHUNK_SIZE = 10 


def parse_hit_object_line(line):
    fields = line.strip().split(",")
    time = np.array([normalize(int(fields[2]),lower_bound=0,upper_bound=LENGTH_THRESHOLD)])
    hit_object_type = np.array(one_hot_encode_object_type(int(fields[3])))
    pos_x = np.array([normalize(int(fields[0]),lower_bound=X_LOWER_BOUND,upper_bound=X_UPPER_BOUND)])
    pos_y = np.array([normalize(int(fields[1]),lower_bound=Y_LOWER_BOUND,upper_bound=Y_UPPER_BOUND)])
    slider_type = np.zeros(4)
    slider_points = np.zeros((MAX_SLIDER_POINTS, 2)) 
    slider_length = np.array([0])
    if hit_object_type[0]:  # Circle
        hit_sound = np.array([int(fields[4])])
    elif hit_object_type[1]:  # Slider

        slider_data = fields[5].split("|")
        slider_type = np.array(one_hot_encode_slider_type(slider_data[0]))
        if len(slider_data)>=2:
            for idx,p in enumerate(slider_data[1:]):
                if idx >= MAX_SLIDER_POINTS:  # limit number of slider points
                    break
                x,y = p.split(":")
                x = normalize(int(x),lower_bound=X_LOWER_BOUND,upper_bound=X_UPPER_BOUND)
                y = normalize(int(y),lower_bound=Y_LOWER_BOUND,upper_bound=Y_UPPER_BOUND)
                slider_points[idx] = [x, y]

        slider_length = np.array([normalize(float(fields[7]),lower_bound=0,upper_bound=SLIDER_LENGTH_THRESHOLD)])
    else:  # osu!mania hold
        pass

    return np.concatenate([time, hit_object_type, pos_x, pos_y, slider_type, slider_points.flatten(), slider_length]).tolist()



def parse_beatmap_file(file_path):
    with open(file_path, "r", encoding="utf8") as f:
        lines = f.readlines()

    hit_objects = []
    hit_object_chunk = []  # List to store a chunk of hit objects
    hit_object_section = False

    for line in lines:
        if line.startswith("[HitObjects]"):
            hit_object_section = True
            continue
        elif line.startswith("[") or line.startswith("\n"):
            timing_point_section = False
            hit_object_section = False

        if hit_object_section:
            hit_object_chunk.append(parse_hit_object_line(line))

            # If the chunk has reached the desired length, append it to the list of chunks
            if len(hit_object_chunk) == CHUNK_SIZE:
                hit_objects.append(np.array(hit_object_chunk))
                hit_object_chunk = []

    # If there are any leftover hit objects, append them as a smaller chunk
    if hit_object_chunk:
        while len(hit_object_chunk) < CHUNK_SIZE:
            hit_object_chunk.append(np.zeros(161))  # add a "padding event"
        hit_objects.append(np.array(hit_object_chunk))

    return np.array(hit_objects)




def chunk_map_data(data,map_hash,chunk_size=120, time_interval=1000):

    time_interval = normalize(time_interval,lower_bound=0,upper_bound=LENGTH_THRESHOLD)

    data = sorted(data, key=lambda x: x[0])

    chunks = []
    current_chunk = []
    first_object_time = data[0][0] # Initialize at first hitobject timing
    padding = [0] * 161

    current_time = 0

    while (first_object_time > current_time + time_interval):
        current_time += time_interval
    

    for hitobject in data:

        if len(current_chunk) == chunk_size:
            print(f"Map chunk capacity was reached once for {map_hash} at {current_time} !")
            
        if hitobject[0] - current_time < time_interval and len(current_chunk) < chunk_size:
            current_chunk.append(hitobject)
        else:
            # Add padding if necessary
            # if len(current_chunk) > 0 and current_chunk[-1][2] == 1:  # If it's a slider
            #     new_chunk_start = [current_chunk[-1]]
            # else:
            #     new_chunk_start = []
            # Add padding if necessary
            while len(current_chunk) < chunk_size:
                current_chunk.append(padding)
            chunks.append(current_chunk)


            while hitobject[0] > current_time + time_interval:
                current_time += time_interval

            current_chunk = []
            if hitobject[0] - current_time < time_interval:
                current_chunk.append(hitobject)

    # Add the last chunk
    while len(current_chunk) < chunk_size:
        current_chunk.append(padding)
    chunks.append(current_chunk)

    # if chunks[0][0][0] == 0:
    #     chunks.pop(0)

    return [e for e in chunks if e[0][0] != 0]







def parse_beatmap_file_flattened(file_path):
    with open(file_path, "r", encoding="utf8") as f:
        lines = f.readlines()

    hit_object_chunk = []  # List to store a chunk of hit objects
    hit_object_section = False

    for line in lines:
        if line.startswith("[HitObjects]"):
            hit_object_section = True
            continue
        elif line.startswith("[") or line.startswith("\n"):
            hit_object_section = False

        if hit_object_section:
            hit_object_chunk.append(parse_hit_object_line(line))

    return hit_object_chunk


def parse_beatmap_file_and_chunk(file_path):
    parsed_datas = parse_beatmap_file_flattened(file_path)
    return chunk_map_data(parsed_datas,file_path)

def get_last_object_timing(beatmap_file):
    if not os.path.exists(beatmap_file):
        print(f"{beatmap_file} file not found")
        return 0 
    
    with open(beatmap_file, 'r', encoding="utf-8") as f:
        lines = [line for line in f.read().split('\n') if line.strip()]
    
    if not lines:
        print(f"No lines found in {beatmap_file}")
        return 0

    fields = lines[-1].split(",")
    return int(fields[2]) if fields else None
     



def one_hot_encode_object_type(object_type):
    # [hit_circle, slider, spinner]
    one_hot = [0, 0, 0]

    base_object_types = [(0, 0), (1, 1), (3, 2)]
    for bit, idx in base_object_types:
        if object_type & (1 << bit):
            one_hot[idx] = 1

    return one_hot

def one_hot_encode_slider_type(slider_type):
    types = ['L', 'P', 'B', 'C']
    encoding = [0]*len(types)
    if slider_type in types:
        encoding[types.index(slider_type)] = 1
    return encoding




# def normalize_ar_cs(value):
#     min_val = 0
#     max_val = 10
#     normalized_value = (value - min_val) / (max_val - min_val)
#     return normalized_value



if __name__ == "__main__":
    beatmap_path = "D:\\osu!rdr_dataset\\beatmaps\\00d6e510453e79f7375bf378a3424f9b.osu"

    beatmap_data = parse_beatmap_file(beatmap_path)
    print(beatmap_data.shape)
import numpy as np
import pandas as pd
from keras.utils import pad_sequences

import beatmap as osu_beatmap
import core as osu_core
import hitobjects as hitobjects
import replay as osu_replay




"""
Ensemble de fonctions permettant de lire un dataset replay/beatmap et de le transformer en données utilisables par le LSTM.

"""




# Constants
BATCH_LENGTH    = 2048
FRAME_RATE      = 24

# Feature index
INPUT_FEATURES  = ['x', 'y','visible', 'is_slider', 'is_spinner']
OUTPUT_FEATURES = ['x', 'y','K1',"K2"]

# Default beatmap frame information
_DEFAULT_BEATMAP_FRAME = (
    osu_core.SCREEN_WIDTH / 2, osu_core.SCREEN_HEIGHT / 2, # x, y
    float("inf"), False, False # time_left, is_slider, is_spinner
)



def load(files):
    """Charge les beatmaps/replay dans leur objects respectifs"""

    replays = []
    beatmaps = []

    for _, row in files.iterrows():

        try:
            replay = osu_replay.load(row['replay'])
            assert not replay.has_mods(osu_replay.Mod.DT, osu_replay.Mod.HR),\
                    "DT and HR are not supported yet"
            beatmap = osu_beatmap.load(row['beatmap'])

        except:
            continue

        replays.append(replay)
        beatmaps.append(beatmap)
        
    return pd.DataFrame(list(zip(replays, beatmaps)), columns=['replay', 'beatmap'])


def input_data(dataset):
    """Transforme un dataset avec les beatmaps en positions au fil du temps (input tensor pour le LSTM)
    """

    data = []
    _memo = {}

    # print(dataset)

    if isinstance(dataset, osu_beatmap.Beatmap):
        dataset = pd.DataFrame([dataset], columns=['beatmap'])


    for beatmap in dataset['beatmap']:

        if beatmap in _memo:
            data += _memo[beatmap]
            continue

        if len(beatmap.hit_objects) == 0:
            continue

        _memo[beatmap] = []
        chunk = []
        preempt, _ = beatmap.approach_rate()
        last_ok_frame = None # Last frame with at least one visible object

        for time in range(beatmap.start_offset(), beatmap.length(), FRAME_RATE):
            frame = _beatmap_frame(beatmap, time)

            if frame is None:
                if last_ok_frame is None:
                    frame = _DEFAULT_BEATMAP_FRAME
                else:
                    frame = list(last_ok_frame)
                    frame[2] = float("inf")
            else:
                last_ok_frame = frame

            px, py, time_left, is_slider, is_spinner = frame

            chunk.append(np.array([
                px - 0.5,
                py - 0.5,
                time_left < preempt,
                is_slider,
                is_spinner
            ]))
            
            if len(chunk) == BATCH_LENGTH:
                data.append(chunk)
                _memo[beatmap].append(chunk)
                chunk = []
        
        if len(chunk) > 0:
            data.append(chunk)  
            _memo[beatmap].append(chunk)

    data = pad_sequences(np.array(data), maxlen=BATCH_LENGTH,
                            dtype='float', padding='post', value=0)
    
    index = pd.MultiIndex.from_product([
        range(len(data)), range(BATCH_LENGTH)
        ], names=['chunk', 'frame'])

    data = np.reshape(data, (-1, len(INPUT_FEATURES)))
    return pd.DataFrame(data, index=index, columns=INPUT_FEATURES, dtype=np.float32)


def target_data(dataset):

    """Transform un dataset de Replay en target tensor pour le LSTM"""

    target_data = []
    
    for replay, beatmap in zip(dataset['replay'],dataset['beatmap']):

        if len(beatmap.hit_objects) == 0:
            continue

        chunk = []

        for time in range(beatmap.start_offset(), beatmap.length(), FRAME_RATE):
            x, y,k1,k2 = _replay_frame(beatmap, replay, time)

            chunk.append(np.array([x-0.5 , y-0.5,k1,k2])) # - 0.5 pour centrer autour de 0

            if len(chunk) == BATCH_LENGTH:
                target_data.append(chunk)
                chunk = []

        if len(chunk) > 0:
            target_data.append(chunk)

    data = pad_sequences(np.array(target_data), maxlen=BATCH_LENGTH, dtype='float', padding='post', value=0)
    index = pd.MultiIndex.from_product([range(len(data)), range(BATCH_LENGTH)], names=['chunk', 'frame'])
    return pd.DataFrame(np.reshape(data, (-1, len(OUTPUT_FEATURES))), index=index, columns=OUTPUT_FEATURES, dtype=np.float32)



def get_key_presses(key_code):
    M1 = 1
    K1 = 4
    M2 = 2
    K2 = 8
    K1_pressed = bool(key_code & (M1 | K1))
    K2_pressed = bool(key_code & (M2 | K2))

    return int(K1_pressed) , int(K2_pressed)




def _beatmap_frame(beatmap, time):
    visible_objects = beatmap.visible_objects(time, count=1)

    if len(visible_objects) > 0:
        obj = visible_objects[0]
        beat_duration = beatmap.beat_duration(obj.time)
        px, py = obj.target_position(time, beat_duration, beatmap['SliderMultiplier'])
        time_left = obj.time - time
        is_slider = int(isinstance(obj, hitobjects.Slider))
        is_spinner = int(isinstance(obj, hitobjects.Spinner))
    else:
        return None

    px = max(0, min(px / osu_core.SCREEN_WIDTH, 1))
    py = max(0, min(py / osu_core.SCREEN_HEIGHT, 1))

    return px, py, time_left, is_slider, is_spinner


def _replay_frame(replay, time):
    x, y, key_code = replay.frame(time)

    key_pressed = get_key_presses(key_code)
    x = max(0, min(x / osu_core.SCREEN_WIDTH, 1))
    y = max(0, min(y / osu_core.SCREEN_HEIGHT, 1))
    return x, y,key_pressed
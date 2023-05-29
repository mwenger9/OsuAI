import os
import numpy as np
from replay_processing import process_replay
from parse_osu_map_file import parse_beatmap_file

def process_and_save_raw_data(raw_data_dir, processed_data_dir, preprocess_function):
    for raw_data_file in os.listdir(raw_data_dir):
        raw_data_filepath = os.path.join(raw_data_dir, raw_data_file)

        processed_data = preprocess_function(raw_data_filepath)

        processed_data_filename = os.path.splitext(raw_data_file)[0] + ".npy"
        processed_data_filepath = os.path.join(processed_data_dir, processed_data_filename)
        np.save(processed_data_filepath, processed_data)


raw_replay_data_dir = "D:\\osu!rdr_dataset\\replays\\osr"
raw_beatmap_data_dir = "D:\\osu!rdr_dataset\\beatmaps"
processed_replay_data_dir = "D:\\osu!rdr_dataset\\replays\\preprocessed"
processed_beatmap_data_dir = "D:\\osu!rdr_dataset\\beatmaps\\prepocessed"

process_and_save_raw_data(raw_replay_data_dir, processed_replay_data_dir, process_replay)
process_and_save_raw_data(raw_beatmap_data_dir, processed_beatmap_data_dir, parse_beatmap_file)

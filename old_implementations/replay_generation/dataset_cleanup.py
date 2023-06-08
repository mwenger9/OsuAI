import pandas as pd
import os
from tqdm import tqdm
import multiprocessing
from parse_osu_map_file import get_last_object_timing
from multiprocessing import Pool



"""
Ensemble de fonctions utilisées pour nettoyer le jeu de données, afin que les données soient assez homogènes (pas d'outliers)

"""


LENGTH_THRESHOLD = 360000 # In milliseconds
STAR_RATING_THRESHOLD = 8
ACCURACY_THRESHOLD = 0.92
SLIDER_LENGTH_THRESHOLD = 3000
SLIDER_POINTS_THRESHOLD = 75

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)




def clean_up(path):

    index = pd.read_csv(f"{path}\\index.csv")

    index['star_rating'] = index['summary'].str.extract(r'\[(.*?) ⭐\]', expand=False)
    index['star_rating'] = index['star_rating'].astype(float)

    index["beatmap_length"] = index["beatmapHash"].apply(lambda beatmapHash : get_last_object_timing(os.path.join(path,beatmapHash)))

    hashes_to_delete = index.loc[(index["mods"] != 0) | (index["star_rating"] > STAR_RATING_THRESHOLD) | (index['performance-Accuracy'] < ACCURACY_THRESHOLD),["beatmapHash","replayHash","mods","summary"]]


    
    index_filtered = index.drop(hashes_to_delete.index)
    index_filtered.to_csv(f"{path}\\index_filtered.csv",mode="w+")


    osr_path = f"{path}\\replays\\osr"

    num_processes = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(processes=num_processes)
    print("deleting osr..")
    results = pool.map_async(delete_file, [f"{osr_path}\\{f}.osr" for f in hashes_to_delete["replayHash"]])
    results.wait()

def count_missing_beatmaps(beatmap_dir,index):
    return len([beatmap for beatmap in index["beatmapHash"] if not os.path.exists(f"{beatmap_dir}\\{beatmap}.osu")])


# def gather_current_max_slider_length(beatmap_file):
#     data = parse_beatmap_file(beatmap_file)
#     slider_length_list = [hobj["Slider Length"] for hobj in data["hit_objects"] if hobj["Slider Length"]]
#     return max(slider_length_list) if slider_length_list else 0


# def clean_unreal_sliders(bm_path,index):
#     currentwd = os.getcwd()
#     os.chdir(bm_path)
#     with multiprocessing.Pool() as p:
#         index["max_slider_length"] = p.map(gather_current_max_slider_length, [beatmapHash+".osu" for beatmapHash in index["beatmapHash"].tolist()])
#     os.chdir(currentwd)

#     index = index[index["max_slider_length"] <= SLIDER_LENGTH_THRESHOLD]


def clean_lengthy_maps(bm_path,index):
    currentwd = os.getcwd()
    os.chdir(bm_path)
    with Pool() as p:
        index["beatmap_length"] = p.map(get_last_object_timing, [beatmapHash+".osu" for beatmapHash in index["beatmapHash"].tolist()])
    os.chdir(currentwd)
    index = index[index['beatmap_length'] <= LENGTH_THRESHOLD]
    # print(len(index[index['beatmap_length'] <= LENGTH_THRESHOLD]))
    index.to_csv('filtered_lengths_index.csv', index=True,header=True)



def gather_current_max_slider_points(beatmap_file):
    # #data = parse_beatmap_file(beatmap_file)
    # slider_points = [hobj["Slider Points"] for hobj in data if hobj["Slider Length"]]
    # return len(max(slider_points,key=len)) if slider_points else 0
    return

def clean_lengthy_slider_points(bm_path,index):
    currentwd = os.getcwd()
    os.chdir(bm_path)
    with multiprocessing.Pool() as p:
        index["max_slider_points"] = p.map(gather_current_max_slider_points, tqdm([beatmapHash+".osu" for beatmapHash in index["beatmapHash"].tolist()]))
    os.chdir(currentwd)

    index = index[index["max_slider_points"] <= SLIDER_POINTS_THRESHOLD]

if __name__ == "__main__":


    path = "D:\\osu!rdr_dataset"
    index = pd.read_csv(f"{path}\\final_cleaned_index.csv")

    index = index[index["playerName"] != "Guest"]
    index = index[index["playerName"] != "Auto+"]
    print(len(index))

    index.to_csv("final_cleaned_index.csv")


    # Checking if the cleanup results





    
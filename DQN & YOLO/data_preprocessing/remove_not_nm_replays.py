import pandas as pd
import os
from tqdm import tqdm
import multiprocessing

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)



if __name__ == "__main__":

    path = "D:\\osu!rdr_dataset"

    index = pd.read_csv(f"{path}\\index.csv")
    hashes_to_delete = index.loc[index["mods"] != 0,["beatmapHash","replayHash","mods"]]
    new_index = index.loc[index["mods"] == 0]
    new_index.to_csv(f"{path}\\cleaned_index.csv",mode="w+")


    osr_path = f"{path}\\replays\\osr"
    npy_path = f"{path}\\replays\\npy"

    bm_path = f"{path}\\beatmaps"

    num_processes = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(processes=num_processes)
    print("deleting osr..")
    results = pool.map_async(delete_file, [f"{osr_path}\\{f}.osr" for f in hashes_to_delete["replayHash"]])
    results.wait()

    print("deleting npy..")
    results = pool.map_async(delete_file, [f"{npy_path}\\{f}.npy" for f in hashes_to_delete["replayHash"]])
    results.wait()

    print("deleting beatmaps..")
    results = pool.map_async(delete_file, [f"{bm_path}\\{f}.osu" for f in hashes_to_delete["beatmapHash"]])
    results.wait()



    # for replayHash in tqdm(hashes_to_delete["replayHash"]):
    #     if os.path.exists(f"{path}\\replays\\osr\\{replayHash}.osr"):
    #         os.remove(f"{path}\\replays\\osr\\{replayHash}.osr")
    #         deleted_osr += 1
        
    #     if os.path.exists(f"{path}\\replays\\npy\\{replayHash}.npy"):
    #         os.remove(f"{path}\\replays\\npy\\{replayHash}.npy")
    #         deleted_npy += 1
        
    
    # for beatmapHash in tqdm(hashes_to_delete["beatmapHash"]):
    #     if os.path.exists(f"{path}\\beatmaps\\{beatmapHash}.osu"):
    #         os.remove(f"{path}\\beatmaps\\{beatmapHash}.osu")
    #         deleted_beatmaps += 1

    # print(f'deleted_osr : {deleted_osr}/{len(hashes_to_delete["replayHash"])}')
    # print(f'deleted_npy : {deleted_npy}/{len(hashes_to_delete["replayHash"])}')
    # print(f'deleted_beatmaps : {deleted_beatmaps}/{len(hashes_to_delete["beatmapHash"])}')
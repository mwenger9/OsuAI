import os
import glob

path = "D:\\osu!rdr_dataset"

# set path to replays folder
replays_path = f"{path}\\replays\\osr"

# set path to beatmaps folder
beatmaps_path = f"{path}/to/beatmaps/folder"

# loop over all beatmap hashes
for beatmap_hash in os.listdir(beatmaps_path):
    
    # check if the file is a .osu file
    if beatmap_hash.endswith(".osu"):
        
        # create a pattern to find the corresponding replay file(s)
        replay_pattern = os.path.join(replays_path, beatmap_hash.replace(".osu", "") + "*.osr")
        print(replay_pattern)
        
        # use glob to find all replay files matching the pattern
        replay_files = glob.glob(replay_pattern)
        print(replay_files)
        
        # loop over all replay files found
        # for replay_file in replay_files:
            
        #     # check if the replay file is for a no-mod play
        #     if "nomod" not in replay_file:
                
        #         # if so, remove the replay file
        #         os.remove(replay_file)

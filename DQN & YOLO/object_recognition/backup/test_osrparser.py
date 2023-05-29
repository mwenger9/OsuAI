from osrparse import Replay
# from a path

path = "D:\\osu!rdr_dataset"
replay_path = f"{path}\\replays\\osr\\0000cb47a2940ce372322c033e4fed90.osr"

replay = Replay.from_path(replay_path)

print(replay.replay_data)
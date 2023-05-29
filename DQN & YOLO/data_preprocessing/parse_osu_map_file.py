from pprint import pprint
from coordinates_utils import osu_pixel_to_screen_coordinates,normalize_coordinates





def parse_timing_point_line(line):
    fields = line.strip().split(',')
    time = int(fields[0])
    beat_length = float(fields[1])
    bpm = 60000 / beat_length
    time_signature = [int(fields[2]), 4] if int(fields[3]) == 0 else [int(fields[2]), int(fields[3])]
    return {
        "Time": time,
        "beatLength": beat_length,
        "Bpm": bpm,
        "TimeSignature": time_signature
    }

def parse_hit_object_line(line):
    fields = line.strip().split(",")
    time = int(fields[2])
    hit_object_type = int(fields[3])
    pos_x = int(fields[0])
    pos_y = int(fields[1])
    pos_x,pos_y = osu_pixel_to_screen_coordinates(pos_x,pos_y)
    pos_x,pos_y = normalize_coordinates(pos_x,pos_y)
    end_time = None
    repeat_count = None
    slider_type = None
    slider_points = None
    slider_length = None
    hit_sound = None
    if hit_object_type & 1:  # Circle
        hit_sound = int(fields[4])
    elif hit_object_type & 2:  # Slider

        hit_sound = int(fields[4])
        slider_data = fields[5].split("|")
        slider_type = slider_data[0]
        slider_points = [tuple(map(int, p.split(":"))) for p in slider_data[1].split(",")]
        slider_length = float(fields[7])
        end_time = time + int(slider_length)
    else:  # osu!mania hold
        pass
    return {
        "Time": time,
        "Type": hit_object_type,
        "Position": (pos_x, pos_y),
        "End Time": end_time,
        "Repeat Count": repeat_count,
        "Slider Type": slider_type,
        "Slider Points": slider_points,
        "Slider Length": slider_length,
        "Hit Sound": hit_sound,
    }




def parse_beatmap_file(file_path) :
    with open(file_path, "r", encoding="utf8") as f:
        lines = f.readlines()

    
    timing_points = []
    hit_objects = []
    beatmap_infos = {}
    timing_point_section = False
    hit_object_section = False

    hit_object_section = False
    for line in lines:
        if line.startswith("ApproachRate:"):
            beatmap_infos["AR"] = line.split(":")[1]

        if line.startswith("CircleSize:"):
            beatmap_infos["CS"] = line.split(":")[1]

        if line.startswith("[TimingPoints]"):
            timing_point_section = True
            continue
        elif line.startswith("[HitObjects]"):
            hit_object_section = True
            continue
        elif line.startswith("[") or line.startswith("\n"):
            timing_point_section = False
            hit_object_section = False

        if timing_point_section:
            timing_points.append(parse_timing_point_line(line))

        if hit_object_section:
            hit_objects.append(parse_hit_object_line(line))

    beatmap_infos["timing_points"] = timing_points
    beatmap_infos["hit_objects"] = hit_objects
    return beatmap_infos


def normalize_ar_cs(value):
    min_val = 0
    max_val = 10
    normalized_value = (value - min_val) / (max_val - min_val)
    return normalized_value


def one_hot_encode_object_type(object_type):
    hit_circle = [1, 0, 0]
    slider = [0, 1, 0]
    spinner = [0, 0, 1]

    if object_type == "hit_circle":
        return hit_circle
    elif object_type == "slider":
        return slider
    elif object_type == "spinner":
        return spinner
    else:
        raise ValueError(f"Invalid object type: {object_type}")


if __name__ == "__main__":
    path = "D:\\osu!rdr_dataset"
    map_path = "D:\\osu!rdr_dataset\\beatmaps\\00010f0eb02ee131aacac54bf72d5444.osu"
    map_data = parse_beatmap_file(map_path)
    print(map_data.keys())

    x,y = map_data["hit_objects"][0]["Position"]
    print(map_data["hit_objects"][0])

import math

# playfield_width=1128, playfield_height = 845, playfield_left=396, playfield_top=158

def osu_pixel_to_screen_coordinates(x_osu, y_osu,playfield_width=1152,playfield_height = 864, playfield_left=383, playfield_top=129):
    osu_playfield_base_width = 512
    osu_playfield_base_height = 384

    x_scaling_factor = playfield_width / osu_playfield_base_width
    y_scaling_factor = playfield_height / osu_playfield_base_height
    x_screen = (x_osu * x_scaling_factor) + playfield_left
    y_screen = (y_osu * y_scaling_factor) + playfield_top

    return x_screen, y_screen


def normalize_coordinates(x_screen, y_screen, playfield_width=1152, playfield_height=864,playfield_left=383, playfield_top=129):
    x_normalized = (x_screen - playfield_left) / playfield_width
    y_normalized = (y_screen - playfield_top) / playfield_height
    return max(0, min(x_normalized, 1)), max(0, min(y_normalized, 1))


def normalize_hit_object_size(size, playfield_width, playfield_height):
    min_size = 0
    max_size = math.sqrt(playfield_width ** 2 + playfield_height ** 2)
    normalized_size = (size - min_size) / (max_size - min_size)
    return normalized_size







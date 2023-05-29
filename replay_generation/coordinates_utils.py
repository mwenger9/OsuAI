import math
import numpy as np
# playfield_width=1128, playfield_height = 845, playfield_left=396, playfield_top=158
X_UPPER_BOUND = 600
X_LOWER_BOUND = -180
Y_LOWER_BOUND = -82
Y_UPPER_BOUND = 407

def normalize(value, lower_bound, upper_bound):
    normalized_value = (value - lower_bound) / (upper_bound - lower_bound)
    return np.float16(normalized_value)








# contains map abstraction functions

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# upsampling by repeating an element in map_in several times to map_out
def upsample(map_in, map_dim_width=40, map_dim_height=40):
    map_height, map_width = map_in.shape
    map_out = np.zeros(shape=(map_dim_height, map_dim_width))
    patch_width = map_dim_width / map_width
    patch_height = map_dim_height / map_height

    for y in range(map_height):
        for x in range(map_width):
            # do sum clampin'
            y_start = int(y * patch_height)
            y_end   = min(int((y + 1) * patch_height), map_dim_height)
            x_start = int(x * patch_width)
            x_end   = min(int((x + 1) * patch_width), map_dim_width)

            map_out[y_start:y_end, x_start:x_end] = map_in[y][x]
    
    return map_out

# downsample by min pooling
def downsample(map_in, map_dim_width=40, map_dim_height=40):
    map_height, map_width = map_in.shape
    map_out = np.zeros(shape=(map_dim_height, map_dim_width))
    patch_width = map_width / map_dim_width
    patch_height = map_height / map_dim_height

    for y in range(map_dim_height):
        for x in range(map_dim_width):
            # do sum clampin'
            y_start = int(y * patch_height)
            y_end   = min(int((y + 1) * patch_height), map_height)
            x_start = int(x * patch_width)
            x_end   = min(int((x + 1) * patch_width), map_width)

            patch = map_in[y_start:y_end, x_start:x_end]
            map_out[y][x] = patch.min()


    return map_out


def load_bmp_to_map(bmp_path, map_dim_width=40, map_dim_height=40):
    bmp_img = Image.open(bmp_path).convert("L")
    bmp_arr = np.array(bmp_img)
    img_height, img_width = bmp_arr.shape

    if (img_width > map_dim_width and img_height > map_dim_height):
        resulting_map = downsample(bmp_arr, map_dim_width, map_dim_height)
    elif (img_width < map_dim_width and img_height < map_dim_height):
        resulting_map = upsample(bmp_arr, map_dim_width, map_dim_height)
    else:
        resulting_map = bmp_arr.astype(float)

    # print(bmp_arr)

    # plt.imshow(resulting_map)
    # plt.show()

    # print(resulting_map.shape)

    return resulting_map

# load_bmp_to_map("./map_bmps/map1.bmp")
# load_bmp_to_map("./map_bmps/map2.bmp")
# load_bmp_to_map("./map_bmps/map3.bmp")
# load_bmp_to_map("./map_bmps/map4.bmp")
# load_bmp_to_map("./map_bmps/spiral.bmp")
# load_bmp_to_map("./map_bmps/hi.bmp")
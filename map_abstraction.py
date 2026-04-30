# contains map abstraction functions

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def upsample(map_in, map_dim_width=40, map_dim_height=40):
    map_width, map_height = map_in.shape

def downsample(map_in, map_dim_width=40, map_dim_height=40):
    map_width, map_height = map_in.shape
    map_out = np.zeros(shape=(map_dim_height, map_dim_width))
    patch_width = map_width // map_dim_width
    patch_height = map_height // map_dim_height
    # print(x_stride)
    # print(y_stride)

    for y in range(map_dim_height):
        for x in range(map_dim_width):
            max_val = -1
            for patch_y in range(patch_height):
                for patch_x in range(patch_width):
                    val = map_in[y*patch_height + patch_y][x * patch_width + patch_x]
                    max_val = max(max_val, val)
                    
            map_out[y][x] = max_val
    
    return map_out


def load_bmp_to_map(bmp_path, map_dim_width=40, map_dim_height=40):
    bmp_img = Image.open(bmp_path)
    bmp_arr = np.array(bmp_img)
    img_width, img_height = bmp_arr.shape

    if (img_width > map_dim_width and img_height > map_dim_height):
        resulting_map = downsample(bmp_arr, map_dim_width, map_dim_height)
    elif (img_width < map_dim_width and img_height < map_dim_height):
        resulting_map = upsample(bmp_arr, map_dim_width, map_dim_height)

    # print(bmp_arr)

    plt.imshow(resulting_map)
    plt.show()

    print(resulting_map.shape)

    pass
    # return map_abstraction (guessing it's a 2d matrix, subsampled accordingly)

# load_bmp_to_map("./map_bmps/map1.bmp")
load_bmp_to_map("./map_bmps/map2.bmp")
load_bmp_to_map("./map_bmps/map3.bmp")
load_bmp_to_map("./map_bmps/map4.bmpgi")
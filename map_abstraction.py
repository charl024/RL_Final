# contains map abstraction functions

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_png_to_map(png_path):

    pass
    # return map_abstraction (guessing it's a 2d matrix, subsampled accordingly)

def upsample(map):
    pass

def downsample(map):
    pass

def load_bmp_to_map(bmp_path, map_dim_width=40, map_dim_height=40):
    bmp_img = Image.open(bmp_path)
    bmp_arr = np.array(bmp_img)
    img_width, img_height = bmp_arr.shape

    if (img_width > map_dim_width and img_height > map_dim_height):
        resulting_map = upsample(bmp_arr)
    elif (img_width < map_dim_width and img_height < map_dim_height):
        resulting_map = downsample(bmp_arr)

    print(bmp_arr)

    plt.imshow(bmp_arr)
    plt.show()

    print(bmp_arr.shape)

    pass
    # return map_abstraction (guessing it's a 2d matrix, subsampled accordingly)

load_bmp_to_map("./map_bmps/map1.bmp")
load_bmp_to_map("./map_bmps/map2.bmp")
load_bmp_to_map("./map_bmps/map3.bmp")
load_bmp_to_map("./map_bmps/map4.bmp")
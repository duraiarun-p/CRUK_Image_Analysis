import glob
import math
import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def stitch_per_Grid_stitching(base_dir, reg_line_init="intensity", tile_size=512):
    exp_name = base_dir.split("/")[-1]
    if os.path.isdir(f"{base_dir}/regrid"):
        base_dir = f"{base_dir}/regrid"
        reg_line_init = "r"

    result_file = open(f"{base_dir}/TileConfiguration.registered.txt", "r")
    lines = result_file.readlines()

    intensity_files = []
    lifetime_files = []
    positions = {}
    max_pos_1 = 0
    max_pos_2 = 0
    min_pos_1 = 10000
    min_pos_2 = 10000
    for line_num, line in enumerate(lines):
        if not line.startswith(reg_line_init):
            continue

        # print(f"{line_num}==={line}")
        temp = line.split(";")
        file_name = temp[0].split(":")[-1].strip()
        if base_dir.find("regrid") >= 0:
            tile_index = file_name[0:file_name.find("_intensity")]
        else:
            tile_index = file_name.split(".")[0].split("_")[-1]
        intensity_files.append(file_name)
        lifetime_files.append(file_name.replace("intensity", "lifetime"))
        position = temp[2].split(":")[-1].strip()
        position = position[1:-1]
        x = round(float(position.split(",")[0]))
        y = round(float(position.split(",")[1]))
        positions[tile_index] = [x, y]
        if max_pos_1 < x:
            max_pos_1 = x
        if max_pos_2 < y:
            max_pos_2 = y
        if min_pos_1 > x:
            min_pos_1 = x
        if min_pos_2 > y:
            min_pos_2 = y

    shift_x = 0
    shift_y = 0
    if min_pos_1 < 0:
        shift_x = 10 - min_pos_1
    if min_pos_2 < 0:
        shift_y = 10 - min_pos_2

    for key, value in positions.items():
        positions[key] = [value[0] + shift_x, value[1] + shift_y]

    tile_file_prefix = file_name.split(".")[0].split("_")[0]
    stitch_intensity = np.zeros([max_pos_2 + tile_size + shift_y, max_pos_1 + tile_size + shift_x], dtype=np.uint16)
    stitch_lifetime = np.zeros([max_pos_2 + tile_size + shift_y, max_pos_1 + tile_size + shift_x], dtype=np.uint16)
    # stitch_results = np.zeros([3500, 3500], dtype=np.uint16)
    for t_i, p in positions.items():
        if base_dir.find("regrid") >= 0:
            intensity = cv2.imread(f"{base_dir}/{t_i}_intensity.tif", -1)
            lifetime = cv2.imread(f"{base_dir}/{t_i}_lifetime.tif", -1)
        else:
            intensity = cv2.imread(f"{base_dir}/{tile_file_prefix}_{exp_name}_intensity_{t_i}.tif", -1)
            lifetime = cv2.imread(f"{base_dir}/{tile_file_prefix}_{exp_name}_lifetime_{t_i}.tif", -1)
        mask = np.multiply((intensity>0), (lifetime>0)).astype(int)
        if np.sum(mask) < 0.05*intensity.shape[0]*intensity.shape[1]:
            continue

        stitch_intensity[p[1]:p[1]+tile_size, p[0]:p[0]+tile_size] = intensity
        stitch_lifetime[p[1]:p[1]+tile_size, p[0]:p[0]+tile_size] = lifetime

    plt.tight_layout()
    plt.subplot(1, 2, 1)
    plt.imshow(stitch_intensity, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(stitch_lifetime, cmap="gray")
    plt.show()

    cv2.imwrite(f"{base_dir}/stitched_intensity.tif", stitch_intensity)
    cv2.imwrite(f"{base_dir}/stitched_lifetime.tif", stitch_lifetime)


if __name__ == "__main__":
    base_dir = "/Users/qiang/Devs/data/WT_iTPA/FLIM_Leica/TMA_iTPA_2C_C06"
    stitch_per_Grid_stitching("/Users/qiang/Devs/data/WT_iTPA/FLIM_Leica/TMA_iTPA_1C_C05/Row-2_Col-3", reg_line_init="Tile", tile_size=512)


"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2507131536
"""

import random
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import json
from datetime import datetime

# np.random.seed(42)
FIXSET = True
DATASET_SIZE = 300

nowtime = datetime.now().strftime("%y%m%d%H%M")
DATASET_SAVE_PATH = os.path.join(os.getcwd(), f'{nowtime}_RELdataset')
META_SAVE_PATH = os.path.join(os.getcwd(), f'{nowtime}_RELmetaList.json')

POINT_MIN_NUM = 5
POINT_MAX_NUM = 10
MARGIN = 0.1
MINSEP = 0.1

MARKER_DICT= {
    "circle": "o",
    "triangle_up": "^",
    "triangle_down": "v",
    "square": "s",
    "diamond": "D",
    "pentagon": "p",
    "hexagon1": "h",
    "hexagon2": "H",
    "x_cross": "X",
    "plus_cross": "P",
    "octagon": "8"
}
COLOR_DICT = {
    "black": "#000000",
    "gray": "#808080",
    "red": "#E74C3C",
    "orange": "#E67E22",
    "yellow": "#F1C40F",
    "green": "#2ECC71",
    "blue": "#3498DB",
    "purple": "#9B59B6",
    "brown": "#A0522D",
    "olive": "#808000"
}
DIRECT_LIST = [
    'lower_left', 
    'lower_right',
    'upper_left',
    'upper_right',
]
POINT_NAME_LIST = [
    'A',
    'B',
    'C', 
    'D', 
    'E', 
    'F', 
    'G', 
    'H', 
    'I', 
    'J', 
]


def save_list2json(
    meta_list, 
    save_filename, 
):
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

    with open(save_filename, "w") as file:
        json.dump(meta_list, file, indent=4, default=convert)

def directer(
    direct, 
    margin, 
    min_sep, 
    bx, by
):
    if direct == 'lower_left':
        valid_x_min = margin
        valid_x_max = bx - min_sep
        valid_y_min = margin
        valid_y_max = by - min_sep

    elif direct == 'lower_right':
        valid_x_min = bx + min_sep
        valid_x_max = 1 - margin
        valid_y_min = margin
        valid_y_max = by - min_sep

    elif direct == 'upper_left':
        valid_x_min = margin
        valid_x_max = bx - min_sep
        valid_y_min = by + min_sep
        valid_y_max = 1 - margin

    elif direct == 'upper_right':
        valid_x_min = bx + min_sep
        valid_x_max = 1 - margin
        valid_y_min = by + min_sep
        valid_y_max = 1 - margin

    else:
        raise ValueError(f"Unknown direction '{direct}'")

    if valid_x_max <= valid_x_min or valid_y_max <= valid_y_min:
        raise ValueError(f"No valid space for direction '{direct}' with given margin/min_sep/bx/by")

    ax = np.random.uniform(valid_x_min, valid_x_max)
    ay = np.random.uniform(valid_y_min, valid_y_max)

    return ax, ay

def choose_direct(
    margin, 
    min_sep, 
    bx, 
    by, 
):
    idx = random.randint(0, len(DIRECT_LIST) - 1)
    key = DIRECT_LIST[idx]
    ax, ay = directer(
        direct=key, 
        margin=margin, 
        min_sep=min_sep, 
        bx=bx, 
        by=by,
    )
    value = (ax, ay)
    return key, value

def get_random_points(
    point_min_num, 
    point_max_num,
):
    point_num = random.randint(point_min_num, point_max_num)
    random.shuffle(POINT_NAME_LIST)

    marker_items = list(MARKER_DICT.items())
    random.shuffle(marker_items)
    shuffled_marker_dict = dict(marker_items)

    color_items = list(COLOR_DICT.items())
    random.shuffle(color_items)
    shuffled_color_dict = dict(color_items)

    points_dict = {}
    gt_list = []
    for idx in range(point_num):
        point_name = POINT_NAME_LIST[idx]

        marker_key = list(shuffled_marker_dict.keys())[idx]
        marker_value = shuffled_marker_dict[marker_key]

        color_key = list(shuffled_color_dict.keys())[idx]
        color_value = shuffled_color_dict[color_key]

        points_dict[point_name] = {
            "marker": marker_value,
            "color": color_value,
        }
        gt_list += [{
            "name": point_name,
            "marker": marker_key,
            "color": color_key,
        }]

    return points_dict, gt_list

def syb2str(
    point_dict,
):
    return {
        'sybVp': f"object {point_dict['name']}",
        'imgVp': f"{point_dict['color']} object",
    }

def promptTem(
    point_a, 
    point_b, 
):
    return f"""
    The figure represents a map with multiple objects. Each object is associated with a name as shown in the figure. Please answer the following multiple-choice question based on the provided information. In which direction is {point_a} relative to {point_b}? Available options:
    A. LowerLeft
    B. LowerRight
    C. UpperLeft
    D. UpperRight.
    """

def gt2prompt(
    gt_list, 
    img_name, 
):
    opt_dict = {
        'lower_left': 'A. LowerLeft', 
        'lower_right': 'B. LowerRight', 
        'upper_left': 'C. UpperLeft', 
        'upper_right': 'D. UpperRight', 
    }

    sybVp_promptTem = promptTem(
        point_a=syb2str(gt_list[0])['sybVp'], 
        point_b=syb2str(gt_list[1])['sybVp'], 
    )
    imgVp_promptTem = promptTem(
        point_a=syb2str(gt_list[0])['imgVp'], 
        point_b=syb2str(gt_list[1])['imgVp'], 
    )
    ans = opt_dict[gt_list[-1]]

    return {
        'img_name': img_name, 
        'sybVp_promptTem': sybVp_promptTem, 
        'imgVp_promptTem': imgVp_promptTem, 
        'ans': ans, 
    }

def is_valid(pt, existing, min_dist):
    return all(np.hypot(pt[0]-ex[0], pt[1]-ex[1]) >= min_dist for ex in existing)

def gen_reldataset(
    dataset_size: int = DATASET_SIZE, 
    fixset: bool = FIXSET, 
):
    print('buliding reldataset...')
    os.makedirs(DATASET_SAVE_PATH, exist_ok=True)

    data_list = []
    for sample_idx in tqdm(range(dataset_size)):
        if fixset:
            np.random.seed(42 + sample_idx)

        positions = {}
        points_dict, gt_list = get_random_points(
            POINT_MIN_NUM, 
            POINT_MAX_NUM, 
        )

        while True:
            bx = np.random.uniform(MARGIN, 1 - MARGIN)
            by = np.random.uniform(MARGIN, 1 - MARGIN)
            positions[list(points_dict.keys())[1]] = (bx, by)
            break

        while True:
            try:
                direct_key, direct_value = choose_direct(
                    margin=MARGIN, 
                    min_sep=MINSEP, 
                    bx=bx, 
                    by=by, 
                )
                break
            except ValueError:
                continue

        gt_list += [direct_key]
        positions[list(points_dict.keys())[0]] = direct_value

        for name in points_dict:
            if name in positions:
                continue
            while True:
                x = np.random.uniform(MARGIN, 1 - MARGIN)
                y = np.random.uniform(MARGIN, 1 - MARGIN)
                if is_valid((x, y), positions.values(), MINSEP):
                    positions[name] = (x, y)
                    break

        fig, ax = plt.subplots(figsize=(8,8))
        for name, attrs in points_dict.items():
            x, y = positions[name]
            ax.scatter(x, y, marker=attrs["marker"], s=300,
                    facecolor=attrs["color"], edgecolor='black', linewidth=1)
            ax.text(x + 0.02, y + 0.02, name, fontsize=12, va='center')

        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        img_name = f"{str(sample_idx).zfill(3)}.png"
        plt.savefig(os.path.join(DATASET_SAVE_PATH, img_name), dpi=200, bbox_inches='tight')
        data_list += [gt2prompt(gt_list, img_name)]
        # plt.show()
        plt.close()


    save_list2json(
        meta_list=data_list, 
        save_filename=META_SAVE_PATH, 
    )
    print(f"\n{DATASET_SAVE_PATH}\n{META_SAVE_PATH}\n")

    return {
        'METADATA_PATH': META_SAVE_PATH,
        'DATA_PATH': DATASET_SAVE_PATH,
    }

if __name__ == "__main__":
    _ = gen_reldataset()












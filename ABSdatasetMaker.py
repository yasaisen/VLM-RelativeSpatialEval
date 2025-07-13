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
DATASET_SAVE_PATH = os.path.join(os.getcwd(), f'{nowtime}_ABSdataset')
META_SAVE_PATH = os.path.join(os.getcwd(), f'{nowtime}_ABSmetaList.json')

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
QUADRANT_LIST = [
    'quadrant_1',  # upper_right
    'quadrant_2',  # upper_left
    'quadrant_3',  # lower_left
    'quadrant_4',  # lower_right
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

def quadrant_positioner(
    quadrant, 
    margin, 
    min_sep, 
    center_x=0.5, 
    center_y=0.5
):
    if quadrant == 'quadrant_1':
        valid_x_min = max(center_x + min_sep, margin)
        valid_x_max = 1 - margin
        valid_y_min = max(center_y + min_sep, margin)
        valid_y_max = 1 - margin

    elif quadrant == 'quadrant_2':
        valid_x_min = margin
        valid_x_max = min(center_x - min_sep, 1 - margin)
        valid_y_min = max(center_y + min_sep, margin)
        valid_y_max = 1 - margin

    elif quadrant == 'quadrant_3':
        valid_x_min = margin
        valid_x_max = min(center_x - min_sep, 1 - margin)
        valid_y_min = margin
        valid_y_max = min(center_y - min_sep, 1 - margin)

    elif quadrant == 'quadrant_4':
        valid_x_min = max(center_x + min_sep, margin)
        valid_x_max = 1 - margin
        valid_y_min = margin
        valid_y_max = min(center_y - min_sep, 1 - margin)

    else:
        raise ValueError(f"Unknown quadrant '{quadrant}'")

    if valid_x_max <= valid_x_min or valid_y_max <= valid_y_min:
        raise ValueError(f"No valid space for quadrant '{quadrant}' with given margin/min_sep")

    px = np.random.uniform(valid_x_min, valid_x_max)
    py = np.random.uniform(valid_y_min, valid_y_max)

    return px, py

def choose_quadrant(
    margin, 
    min_sep, 
    center_x=0.5, 
    center_y=0.5
):
    idx = random.randint(0, len(QUADRANT_LIST) - 1)
    quadrant_key = QUADRANT_LIST[idx]
    px, py = quadrant_positioner(
        quadrant=quadrant_key, 
        margin=margin, 
        min_sep=min_sep, 
        center_x=center_x, 
        center_y=center_y,
    )
    return quadrant_key, (px, py)

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
    point_name, 
):
    return f"""
    The figure represents a map with multiple objects. Each object is associated with a name as shown in the figure. Please answer the following multiple-choice question based on the provided information. Which direction is {point_name} located in the image? Available options:
    A. UpperRight
    B. UpperLeft
    C. LowerLeft
    D. LowerRight.
    """

def gt2prompt(
    img_name, 
    quadrant_info,
):
    opt_dict = {
        'quadrant_1': 'A. UpperRight', 
        'quadrant_2': 'B. UpperLeft', 
        'quadrant_3': 'C. LowerLeft', 
        'quadrant_4': 'D. LowerRight', 
    }

    target_point = quadrant_info['target_point']
    target_quadrant = quadrant_info['quadrant']

    sybVp_promptTem = promptTem(
        point_name=syb2str(target_point)['sybVp'], 
    )
    imgVp_promptTem = promptTem(
        point_name=syb2str(target_point)['imgVp'], 
    )
    ans = opt_dict[target_quadrant]

    return {
        'img_name': img_name, 
        'sybVp_promptTem': sybVp_promptTem, 
        'imgVp_promptTem': imgVp_promptTem, 
        'ans': ans, 
    }

def is_valid(pt, existing, min_dist):
    return all(np.hypot(pt[0]-ex[0], pt[1]-ex[1]) >= min_dist for ex in existing)

def gen_absdataset(
    dataset_size: int = DATASET_SIZE, 
    fixset: bool = FIXSET, 
):
    print('buliding absdataset...')
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

        target_point_name = random.choice(list(points_dict.keys()))
        target_point_info = None
        for point_info in gt_list:
            if point_info['name'] == target_point_name:
                target_point_info = point_info
                break

        quadrant_key, quadrant_position = choose_quadrant(
            margin=MARGIN, 
            min_sep=MINSEP, 
        )
        
        positions[target_point_name] = quadrant_position

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
        
        # ax.axhline(y=0.5, color='lightgray', linestyle='--', alpha=0.5)
        # ax.axvline(x=0.5, color='lightgray', linestyle='--', alpha=0.5)
        
        for name, attrs in points_dict.items():
            x, y = positions[name]
            size = 400 if name == target_point_name else 300
            ax.scatter(x, y, marker=attrs["marker"], s=size,
                    facecolor=attrs["color"], edgecolor='black', linewidth=1)
            ax.text(x + 0.02, y + 0.02, name, fontsize=12, va='center')

        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        quadrant_info = {
            'target_point': target_point_info,
            'quadrant': quadrant_key,
        }

        img_name = f"{str(sample_idx).zfill(3)}.png"
        plt.savefig(os.path.join(DATASET_SAVE_PATH, img_name), dpi=200, bbox_inches='tight')
        data_list += [gt2prompt(img_name, quadrant_info)]
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
    _ = gen_absdataset()











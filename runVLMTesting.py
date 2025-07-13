"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2507131536
"""

import os
import openai
import base64
import os
from datetime import datetime
from tqdm import tqdm
import json

from RELdatasetMaker import gen_reldataset, save_list2json
from ABSdatasetMaker import gen_absdataset


openai.api_key = ""  # TODO: Replace with your api key


def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data

def prompter(
    prompt
):
    target_str = 'Available options:'
    add_str = ' (tips: Please first determine the positions of the two objects on the map, and then identify their relative positions.) '

    return prompt.replace(target_str, add_str + target_str)

def run_vlm_inference(prompt, img_path):
    with open(img_path, "rb") as image_file:
        base64_img = base64.b64encode(image_file.read()).decode("utf-8")

    messages = [
        {"role": "system", "content": "You are a helpful visual assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
        ]}
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14", # Run GPT-4o
            messages=messages,
            max_tokens=512
        )
        content = response.choices[0].message.content.strip()

        return {"answer": content}

    except Exception as e:
        print("Error during inference:", e)
        return {"answer": "", "reasoning": ""}

def run_test(
    test_setting_name: str,
    metadata_path: str, 
    data_path: str, 
    test_vp: str,
    prompter = None,
):
    nowtime = datetime.now().strftime("%y%m%d%H%M")
    print(f'\ntesting {test_setting_name} ({nowtime})...')
    meta_list = load_json_data(metadata_path)

    test_result_list = []
    for idx in tqdm(range(len(meta_list))):

        img_path = os.path.join(data_path, meta_list[idx]['img_name'])
        prompt = meta_list[idx][f"{test_vp}_promptTem"]

        if prompter is not None:
            prompt = prompter(prompt)

        out = run_vlm_inference(
            prompt=prompt,
            img_path=img_path
        )

        test_result_list += [{
            'ans': out["answer"],
            'gt': meta_list[idx]['ans']
        }]

    acc_list = []
    for sample in test_result_list:
        if sample['gt'] in sample['ans']:
            acc_list += [1]
        else:
            acc_list += [0]
    print(f"\nacc: {sum(acc_list) / len(acc_list)}")

    result_name = f"testResult_{nowtime}_{test_setting_name}""""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2506130419
"""
    print(result_name)

def run_testingsets(
    setting_list: list
):
    print('\nstart run_testingsets...\n')
    for sample in setting_list:
        run_test(
            test_setting_name=sample['test_setting_name'],
            metadata_path=sample['metadata_path'],
            data_path=sample['data_path'],
            test_vp=sample['test_vp'],
            prompter=sample['prompter'],
        )
    print('\n...end run_testingsets\n')

if __name__ == "__main__":
    rel_path_dict = gen_reldataset(dataset_size=5)
    abs_path_dict = gen_absdataset(dataset_size=5)

    setting_list = [
        {
            'test_setting_name': 'rel_sybVp_nP', 
            'metadata_path': rel_path_dict['METADATA_PATH'], 
            'data_path': rel_path_dict['DATA_PATH'], 
            'test_vp': 'sybVp', 
            'prompter': None, 
        },
        {
            'test_setting_name': 'rel_imgVp_nP', 
            'metadata_path': rel_path_dict['METADATA_PATH'], 
            'data_path': rel_path_dict['DATA_PATH'], 
            'test_vp': 'imgVp', 
            'prompter': None, 
        },
        {
            'test_setting_name': 'abs_sybVp_nP', 
            'metadata_path': abs_path_dict['METADATA_PATH'], 
            'data_path': abs_path_dict['DATA_PATH'], 
            'test_vp': 'sybVp', 
            'prompter': None, 
        },
        {
            'test_setting_name': 'abs_imgVp_nP', 
            'metadata_path': abs_path_dict['METADATA_PATH'], 
            'data_path': abs_path_dict['DATA_PATH'], 
            'test_vp': 'imgVp', 
            'prompter': None, 
        },
        {
            'test_setting_name': 'rel_sybVp_aP', 
            'metadata_path': rel_path_dict['METADATA_PATH'], 
            'data_path': rel_path_dict['DATA_PATH'], 
            'test_vp': 'sybVp', 
            'prompter': prompter, 
        },
        {
            'test_setting_name': 'rel_imgVp_aP', 
            'metadata_path': rel_path_dict['METADATA_PATH'], 
            'data_path': rel_path_dict['DATA_PATH'], 
            'test_vp': 'imgVp', 
            'prompter': prompter, 
        },
    ]

    run_testingsets(
        setting_list=setting_list,
    )
    












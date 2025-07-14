# VLM-RelativeSpatialEval
`VLM-RelativeSpatialEval` provides a dataset and testing workflow designed to evaluate Visual Language Models (VLMs) in determining relative and absolute spatial positions.

This project comprises three primary Python scripts:
* **ABSdatasetMaker.py**: Generates datasets for absolute position evaluation.
* **RELdatasetMaker.py**: Generates datasets for relative position evaluation.
* **runVLMTesting.py**: Calls the above datasets to conduct testing.
Datasets are output as images along with corresponding JSON files. During the testing phase, an OpenAI API key is required to invoke the model for inference.

## Dataset Design
### Relative Position Dataset (`RELdatasetMaker.py`)
* Each image randomly generates between 5 to 10 labeled points, each assigned a random `color`, `label`, and `shape`.
* Two points are randomly selected as reference and target points, with the target placed in one of the four directions relative to the reference (`lower-left`, `lower-right`, `upper-left`, `upper-right`).
* The resulting JSON records the `filename` for each image, two prompt templates (`symbolic viewpoint` and `image viewpoint`), and the `correct answer` (multiple-choice).
* The function `gen_reldataset()` returns paths for `storing data path` and `JSON path`, with options to customize `dataset size` and `random seed`.
  As shown in the following example

<div align="center">
  <img src="https://github.com/yasaisen/VLM-RelativeSpatialEval/blob/main/doc/rel_000.png" alt="inference" width="300">
</div>

```python
{
    "img_name": "000.png",
    "sybVp_promptTem": "\n    The figure represents a map with multiple objects. Each object is associated with a name as shown in the figure. Please answer the following multiple-choice question based on the provided information. In which direction is object G relative to object C? Available options:\n    A. LowerLeft\n    B. LowerRight\n    C. UpperLeft\n    D. UpperRight.\n    ",
    "imgVp_promptTem": "\n    The figure represents a map with multiple objects. Each object is associated with a name as shown in the figure. Please answer the following multiple-choice question based on the provided information. In which direction is gray object relative to brown object? Available options:\n    A. LowerLeft\n    B. LowerRight\n    C. UpperLeft\n    D. UpperRight.\n    ",
    "ans": "B. LowerRight"
}
```

### Absolute Position Dataset (`ABSdatasetMaker.py`)
* Each image randomly generates between 5 to 10 labeled points, each assigned a random `color`, `label`, and `shape`.
* A target point is randomly selected and placed in one of the four quadrants (`upper-right`, `upper-left`, `lower-left`, `lower-right`).
* The resulting JSON records the `filename` for each image, two prompt templates (`symbolic viewpoint` and `image viewpoint`), and the `correct answer` (multiple-choice).
* The function `gen_absdataset()` returns paths for `storing data path` and `JSON path`, with options to customize `dataset size` and `random seed`.
  As shown in the following example

<div align="center">
  <img src="https://github.com/yasaisen/VLM-RelativeSpatialEval/blob/main/doc/abs_000.png" alt="inference" width="300">
</div>

```python
{
    "img_name": "000.png",
    "sybVp_promptTem": "\n    The figure represents a map with multiple objects. Each object is associated with a name as shown in the figure. Please answer the following multiple-choice question based on the provided information. Which direction is object D located in the image? Available options:\n    A. UpperRight\n    B. UpperLeft\n    C. LowerLeft\n    D. LowerRight.\n    ",
    "imgVp_promptTem": "\n    The figure represents a map with multiple objects. Each object is associated with a name as shown in the figure. Please answer the following multiple-choice question based on the provided information. Which direction is olive object located in the image? Available options:\n    A. UpperRight\n    B. UpperLeft\n    C. LowerLeft\n    D. LowerRight.\n    ",
    "ans": "D. LowerRight"
}
```

### Prompt Templates and Viewpoints
* **sybVp (symbolic viewpoint)**: Describes points by labels, e.g., "object A".
* **imgVp (image viewpoint)**: Describes points by colors, e.g., "red object".
* Each image in the generated dataset includes two types of prompts, facilitating comparisons of their effects on model performance.

## Testing Workflow (`runVLMTesting.py`)
1. Calls `gen_reldataset()` and `gen_absdataset()` to generate datasets.
2. Defines multiple test configurations (`relative/absolute`, `sybVp/imgVp`,` with/without additional prompter`).
3. Performs inference on each image using the OpenAI API and computes accuracy by checking if the response includes the correct option (`exact-match metrics`).
4. Results are displayed in the terminal and saved in a file named `testResult_<datetime>_<configuration_name>`.
   Additional configurations can be tested by modifying the `setting_list`.

The additional prompt inserted is:
```python
"(tips: Please first determine the positions of the two objects on the map, and then identify their relative positions.)"
```
This aims to guide the model to first determine the absolute positions of two points and subsequently infer their relative positions based on those determinations.

## Test Results
Tests were conducted using the `gpt-4.1-nano-2025-04-14` model, with 300 samples evaluated using `exact-match metrics`.

| Configuration  | Test     | Viewpoint | Prompter | Result |
| -------------- | -------- | --------- | -------- | ------ |
| rel\_sybVp\_nP | Relative | Label     | X        | 79.66  |
| rel\_imgVp\_nP | Relative | Color     | X        | 67.00  |
| abs\_sybVp\_nP | Absolute | Label     | X        | 86.00  |
| abs\_imgVp\_nP | Absolute | Color     | X        | 86.33  |
| rel\_sybVp\_aP | Relative | Label     | V        | 83.00  |
| rel\_imgVp\_aP | Relative | Color     | V        | 72.66  |

The results indicate that providing prompts to guide the model in first determining absolute positions significantly assists in improving relative position judgments.

## Installation
1. Create a Python environment (tested with Python 3.10):
   ```bash
   conda create --name vlmRelSpaEval python=3.10
   conda activate vlmRelSpaEval
   ```
2. Clone the repository:
   ```bash
   git clone https://github.com/yasaisen/VLM-RelativeSpatialEval.git
   cd VLM-RelativeSpatialEval
   ```
3. Install the dependencies:
   ```bash
   pip install openai matplotlib tqdm
   ```

## Usage Instructions
1. Enter a valid `openai.api_key` in `runVLMTesting.py`.
2. Run the testing script:
   ```bash
   python runVLMTesting.py  
   ```
   The generated data and annotation JSON will be saved in the current directory, with the filename based on the date and time.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.









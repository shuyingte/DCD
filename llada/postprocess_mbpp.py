# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Adapted from Dream repos for MBPP

import evaluate as hf_evaluate
import os
import sys
import re
from sanitize import sanitize

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
pass_at_k = hf_evaluate.load("code_eval")

def pass_at_1(references, predictions):
    return pass_at_k.compute(
        references=references,
        predictions=predictions,
        k=[1],
    )[0]["pass@1"]

import json

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def extract_entry_point(target_str):
    # target looks like: "assert func_name(...) == ...\nassert ..."
    match = re.search(r'assert\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', target_str)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Could not extract function name from target: {target_str}")

file_path = sys.argv[1]
data = read_jsonl(file_path)

references = [sample['target'] for sample in data]

predictions = []
for sample in data:
    # Extract raw generation
    raw_gen = sample['resps'][0][0]
    # Remove ```python and ``` if present
    code = raw_gen.split('```python\n', 1)[-1].split('```', 1)[0]
    # Extract function name from target
    entry_point = extract_entry_point(sample['target'])
    # Sanitize only the code (no prompt prepended)
    clean_code = sanitize(code, entry_point)
    predictions.append([clean_code])

pass_at_1s = [pass_at_1([reference], [prediction]) for reference, prediction in zip(references, predictions)]

result = sum(pass_at_1s) / len(pass_at_1s)
print(result)
result_file = os.path.join(os.path.split(file_path)[0], 'result.txt')
with open(result_file, 'w') as file:
    file.write(str(result))

def write_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

res = [{"task_id": sample['doc']['task_id'], "completion": pred[0], "pass_at_1": res_val} 
       for sample, pred, res_val in zip(data, predictions, pass_at_1s)]
write_jsonl(res, file_path + '.cleaned')
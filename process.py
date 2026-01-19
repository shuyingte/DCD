import os
import shutil
import time
import json
import glob
import concurrent.futures
import random
import tqdm
import sys
import statistics
import pandas as pd


def _together(dir_path, out_dir):
    # Merge all jsonl files under dir_path
    os.makedirs(out_dir, exist_ok=True)
    files = glob.glob(f"{dir_path}/**/*.json*", recursive=True)
    if all("multicard_" in file for file in files):
        files = sorted(files, key=lambda x: int(os.path.basename(x).split("multicard_")[1].split(".")[0]))
    for file in files:
        out_file = f"{out_dir}/{os.path.basename(dir_path)}.json"
        with open(out_file, "ab") as outf:
            with open(file, "rb") as inf:
                outf.write(inf.read())


def _to_open_compass(file, out_dir, add_gold=False):
    os.makedirs(out_dir, exist_ok=True)
    classed_name = {
        "gsm8k": False,
        "humaneval": False,
        "math_500": False,
        "sanitized_mbpp": False,
        "IFEval": False,
    }
    data_list = pd.read_json(file, lines=True).to_dict(orient="records")

    data_name = None
    classed = False
    for name in classed_name:
        if name in os.path.basename(file):
            data_name = name
            classed = classed_name[name]
            break

    if data_name is None:
        raise ValueError(f"Unrecognized dataset in filename: {file}")

    print(data_name)

    out_dict_dict = dict()
    if add_gold:
        to_datasets = {
            "gsm8k": r"./data/gsm8k_0shot_gen_a58960.json",
            "humaneval": r"./data/humaneval_gen_8e312c.json",
            "math_500": r"./data/math_500_opencompass.json",
            "sanitized_mbpp": r"./data/sanitized_mbpp_mdblock_0shot_nocot_gen_a2e416.json",
            "IFEval": r"./data/IFEval_gen_353ae7.json",
        }
        new_data_list = pd.read_json(to_datasets[data_name], lines=True).to_dict(orient="records")
        for new_data in new_data_list:
            for data in data_list:
                if new_data["prompt"].strip() in data["prompt"]:
                    new_data["response"] = data["response"]
                    break
            assert new_data.get("response", None) is not None, "Failed to match response to prompt"
        data_list = new_data_list

    for data in data_list:
        classes = data["classes"] if classed else "<null>"
        gold = data.get("gold", None)
        if gold is None:
            gold = data.get("gt", None)
        assert gold, f"Missing ground truth in {os.path.basename(file)}"

        drop_think = True
        if drop_think:
            if any(ele in data["response"] for ele in ["[unused17]", "[unused16]"]):
                end_tokens = "[unused17]"
                start_tokens = "[unused16]"
            elif any(ele in data["response"] for ele in ["</think>", "<think>"]):
                end_tokens = "</think>"
                start_tokens = "<think>"
            else:
                drop_think = False
        data["response"] = data["response"].split(end_tokens)[-1] if drop_think else data["response"]
        data["response"] = data["response"].strip("[unused10]")

        out_dict = {
            "origin_prompt": [
                {
                    "role": "HUMAN",
                    "prompt": data["prompt"],
                }
            ],
            "prediction": data["response"],
            "gold": gold
        }
        if classes in out_dict_dict:
            out_dict_dict[classes][len(out_dict_dict[classes])] = out_dict
        else:
            out_dict_dict[classes] = {"0": out_dict}

    for classes, out_dict in out_dict_dict.items():
        if classes == "<null>":
            if data_name == "humaneval":
                out_path = os.path.join(out_dir, f"openai_{data_name}.json")
            elif data_name == "math_500":
                out_path = os.path.join(out_dir, f"math-500.json")
            else:
                out_path = os.path.join(out_dir, f"{data_name}.json")
        else:
            out_path = os.path.join(out_dir, f"{data_name}-{classes}.json")

        with open(out_path, "w", encoding="utf-8") as outf:
            json.dump(out_dict, outf, ensure_ascii=False)


if __name__ == "__main__":
    """
    Main logic consists of five steps:
    1. Copy inference results from a local source path to ./infer_downloads/{save_name}/downloads/
    2. Merge JSONL files within each dataset folder.
    3. Convert merged files into OpenCompass evaluation format (only for specified datasets).
    4. Copy formatted files to OpenCompass prediction directory and run evaluation.
    5. Read and print scores from the latest summary CSV files.
    """

    # 1. Local copy instead of S3 download
    if len(sys.argv) < 3:
        print("Usage: python script.py <source_path> <save_name>")
        sys.exit(1)

    source_path = sys.argv[1]
    save_name = sys.argv[2]

    local_save_dir = r"./infer_downloads/"
    local_save_path = os.path.join(local_save_dir, save_name)
    download_target = os.path.join(local_save_path, "downloads")
    os.makedirs(download_target, exist_ok=True)

    # Copy all contents from source_path to download_target
    if os.path.isdir(source_path):
        for item in os.listdir(source_path):
            s = os.path.join(source_path, item)
            d = os.path.join(download_target, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
    else:
        print(f"Source path is not a directory: {source_path}")
        sys.exit(1)

    download_list = [local_save_path]

    # 2. Merge files
    merge_list = []
    for local_save_path in download_list:
        merge_dir = os.path.join(local_save_path, "merge_files")
        file_dir_list = glob.glob(f"{local_save_path}/downloads/*")
        for file_dir in file_dir_list:
            _together(file_dir, merge_dir)
        merge_list.append(merge_dir)

    # 3. Convert to OpenCompass format
    open_compass_list = []
    for merge_dir in merge_list:
        open_compass_dir = os.path.join(os.path.dirname(merge_dir), "opencompass_file")
        for json_file in glob.glob(f"{merge_dir}/*.json"):
            try:
                _to_open_compass(json_file, open_compass_dir, add_gold=True)
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
        open_compass_list.append(open_compass_dir)

    # 4. Copy to OpenCompass predictions folder and run evaluation
    for open_compass_dir in open_compass_list:
        model_name = os.path.basename(os.path.dirname(open_compass_dir))
        to_dir = os.path.join(r"./outputs/default/predictions/predictions", f"{model_name}_hf")
        os.makedirs(to_dir, exist_ok=True)
        for file in glob.glob(f"{open_compass_dir}/*.json"):
            to_path = os.path.join(to_dir, os.path.basename(file))
            shutil.copyfile(file, to_path)

        datasets_list = [
            'gsm8k_0shot_gen_a58960',
            'humaneval_gen_8e312c',
            'math_500_gen_my',
            'sanitized_mbpp_mdblock_0shot_nocot_gen_a2e416',
            'IFEval_gen_353ae7',
        ]

        def _run(args):
            datasets_name, model_name = args
            random_time = random.uniform(0, 3)
            time.sleep(random_time)
            if any(datasets_name.split("_")[0].lower() in os.path.basename(f).lower() for f in glob.glob(f"{to_dir}/*.json")):
                print("-------- Running command: ---------")
                print(f"python run.py --datasets {datasets_name} --hf-type chat --hf-path {model_name} -w outputs/default -r --mode eval")
                print("----------------------------------")
                os.system(f"python run.py --datasets {datasets_name} --hf-type chat --hf-path {model_name} -w outputs/default -r --mode eval")

        mult_thread_list = [(datasets_name, model_name) for datasets_name in datasets_list]
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(mult_thread_list))) as executor:
            executor.map(_run, mult_thread_list)

    # 5. Read scores from Excel
    excel_list = glob.glob(r"./outputs/default/predictions/summary/*.csv")
    excel_list = sorted(excel_list, key=lambda x: os.path.getmtime(x), reverse=True)[:len(mult_thread_list)]

    for excel_path in excel_list:
        df = pd.read_csv(excel_path, header=None)
        e_column = df.iloc[:, -1].tolist()
        e_data = e_column[1:]
        e_floats = [float(x) for x in e_data if x != '-']
        datasets_name = df.iloc[1, 0].split('-')[0]
        if e_floats:
            if datasets_name not in ["sanitized_mbpp"]:
                average = statistics.mean(e_floats) if len(e_floats) > 1 else e_floats[0]
            else:
                average = e_floats[0]
            print(f"{df.iloc[0, -1]}: \t{datasets_name}: \t{average:.2f}")
import json
import re
import os
import glob
import argparse


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return string
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]
    return retval


def remove_boxed(s):
    if s is None:
        return None
    if "\\boxed " in s:
        left = "\\boxed "
        return s[len(left) :]
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except (AssertionError, IndexError):
        return s


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            # 🔒 关键修复：检查 substr 是否为空
            if not substr:  # 空字符串直接追加
                continue
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    # 如果长度不够，直接追加原 substr（不修复）
                    new_str += substr
                    continue
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if not split:  # 防空
            new_string += "\\sqrt"
            continue
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def fix_a_slash_b(string):
    if string is None:
        return None
    parts = string.split("/")
    if len(parts) != 2:
        return string
    a, b = parts[0], parts[1]
    try:
        float(a)
        float(b)
        return f"\\frac{{{a}}}{{{b}}}"
    except (ValueError, TypeError):
        return string


def strip_string(string):
    if string is None:
        return None
    try:
        string = str(string).strip()
        while re.search(r"(\d),(\d{3})", string):
            string = re.sub(r"(\d),(\d{3})", r"\1\2", string)
        string = string.replace("\n", "")
        string = string.replace("\\!", "")
        string = string.replace("\\\\", "\\")
        string = string.replace("tfrac", "frac").replace("dfrac", "frac")
        string = string.replace("\\left", "").replace("\\right", "")
        string = string.replace("^{\\circ}", "").replace("^\\circ", "")
        string = string.replace("\\$", "").replace("\\%", "").replace("\%", "")
        string = remove_right_units(string)
        if string.startswith("."):
            string = "0" + string
        if " ." in string:
            string = string.replace(" .", " 0.")
        if len(string.split("=")) == 2:
            lhs = string.split("=")[0].strip()
            if len(lhs) <= 2:
                string = string.split("=")[1].strip()
        string = fix_sqrt(string)
        string = string.replace(" ", "")
        string = fix_fracs(string)
        if string == "0.5":
            string = "\\frac{1}{2}"
        string = fix_a_slash_b(string)
        return string
    except Exception:
        # 如果清洗过程崩溃，返回原始字符串（由上层决定是否算错）
        return str(string).strip() if string is not None else ""


def is_equiv(str1, str2, verbose=False):
    if str1 is None or str2 is None:
        return False
    try:
        str1_clean = strip_string(str1)
        str2_clean = strip_string(str2)
    except Exception:
        return False

    # 数值比较
    try:
        if abs(float(str1_clean) - float(str2_clean)) < 1e-6:
            return True
    except (ValueError, TypeError, OverflowError):
        pass

    # 字符串比较
    try:
        return str1_clean == str2_clean
    except Exception:
        return str(str1).strip() == str(str2).strip()


def parse_math500_answers_from_jsonl(json_path):
    total_correct = 0
    total_processed = 0

    with open(json_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                # 无效 JSON，跳过（视为未答对）
                total_processed += 1
                continue

            total_processed += 1
            try:
                ground_truth_str = str(item.get("target", "")).strip()
                raw_generation = ""
                resps = item.get("resps")
                if resps and isinstance(resps, list) and len(resps) > 0 and isinstance(resps[0], list) and len(resps[0]) > 0:
                    raw_generation = str(resps[0][0])

                extracted_answer_str = None
                boxed_str = last_boxed_only_string(raw_generation)
                if boxed_str:
                    extracted_answer_str = remove_boxed(boxed_str)
                if extracted_answer_str is None:
                    answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
                    if answer_match:
                        extracted_answer_str = answer_match.group(1).strip()

                is_correct = is_equiv(extracted_answer_str, ground_truth_str)
                if is_correct:
                    total_correct += 1

            except Exception as e:
                # 任何处理异常都视为该题答错，但继续处理
                # 可选：记录警告（此处省略以保持简洁）
                continue

    return total_correct, total_processed


def evaluate_math500_results(directory):
    jsonl_files = glob.glob(os.path.join(directory, "*.jsonl"))
    if not jsonl_files:
        print(f"Error: No .jsonl files found in directory '{directory}'.")
        return

    total_correct = 0
    total_processed = 0

    for file_path in jsonl_files:
        correct, processed = parse_math500_answers_from_jsonl(file_path)
        total_correct += correct
        total_processed += processed

    if total_processed == 0:
        print("0.00%")
        return

    accuracy = (total_correct / total_processed) * 100
    print(f"{accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--res_path",
        type=str,
        required=True,
        help="Path to the directory containing .jsonl result files"
    )
    args = parser.parse_args()
    evaluate_math500_results(directory=args.res_path)
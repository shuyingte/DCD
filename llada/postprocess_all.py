#!/usr/bin/env python3
"""
Process all JSONL files in a directory recursively for both code and math tasks.
Usage: python llada/postprocess_all.py [directory_path]
"""

import os
import sys
import glob
import subprocess

def find_qualified_directories(target_dir):
    """
    Recursively find all directories that qualify for processing:
    - Exactly one .jsonl file
    - No result.txt exists
    Returns list of (dir_path, jsonl_file_path)
    """
    qualified_dirs = []
    for root, dirs, files in os.walk(target_dir):
        jsonl_files = list(glob.glob(os.path.join(root, "*.jsonl")))
        if len(jsonl_files) != 1:
            continue
        if os.path.exists(os.path.join(root, "result.txt")):
            continue
        qualified_dirs.append((root, jsonl_files[0]))
    return qualified_dirs

def main():
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = '.'

    if not os.path.exists(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist")
        sys.exit(1)
    if not os.path.isdir(target_dir):
        print(f"Error: '{target_dir}' is not a directory")
        sys.exit(1)

    target_dir = os.path.abspath(target_dir)
    print(f"Processing directory recursively: {target_dir}")

    # Get current script dir to locate sibling scripts
    script_dir = os.path.dirname(os.path.abspath(__file__))
    postprocess_humaneval_script = os.path.join(script_dir, "postprocess_code.py")
    postprocess_mbpp_script = os.path.join(script_dir, "postprocess_mbpp.py")
    math500_script = os.path.join(script_dir, "..", "tasks_pro", "math500.py")
    math500_script = os.path.abspath(math500_script)

    # Validate math500 script exists
    if not os.path.exists(math500_script):
        print(f"Warning: math500.py not found at {math500_script}. Math500 tasks will fail.")

    qualified_dirs = find_qualified_directories(target_dir)
    print(f"Found {len(qualified_dirs)} qualified directories")

    stats = {'processed': 0, 'failed': 0, 'ignored': 0}
    errors = []

    for dir_path, jsonl_file in qualified_dirs:
        dir_lower = dir_path.lower()
        try:
            if "humaneval" in dir_lower or 'mbpp' in dir_lower:
                # Code task: call postprocess_code.py <jsonl_file>
                postprocess_code_script = postprocess_humaneval_script if  "humaneval" in dir_lower else postprocess_mbpp_script
                cmd = [sys.executable, postprocess_code_script, jsonl_file]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    # Assume postprocess_code.py already writes result.txt
                    # If not, you may need to write it here — but per your spec, we assume it does.
                    stats['processed'] += 1
                else:
                    stats['failed'] += 1
                    errors.append((dir_path, f"Code task failed (rc={result.returncode})"))
            elif "math500" in dir_lower:
                # Math task: call math500.py -r <dir_path>, capture stdout, write to result.txt
                cmd = [sys.executable, math500_script, "-r", dir_path]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if result.returncode == 0:
                    # Write stdout (e.g., "82.60%") to result.txt
                    result_txt_path = os.path.join(dir_path, "result.txt")
                    with open(result_txt_path, 'w', encoding='utf-8') as f:
                        f.write(result.stdout.strip() + '\n')
                    stats['processed'] += 1
                else:
                    stats['failed'] += 1
                    errors.append((dir_path, f"Math500 task failed (rc={result.returncode})"))
            else:
                # Ignore other tasks
                stats['ignored'] += 1
                continue
        except subprocess.TimeoutExpired:
            stats['failed'] += 1
            errors.append((dir_path, "Subprocess timeout"))
        except Exception as e:
            stats['failed'] += 1
            errors.append((dir_path, str(e)))

    # Output summary
    if errors:
        print("\nFailed tasks:")
        for d, err in errors:
            print(f"  {d}: {err}")

    print(f"\nStatistics:")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Ignored (unknown task): {stats['ignored']}")
    print(f"Total qualified directories: {stats['processed'] + stats['failed'] + stats['ignored']}")

if __name__ == "__main__":
    main()
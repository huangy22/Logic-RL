""" Preprocess dataset for integer add logic task """

import argparse
import json
import os

from datasets import Dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import random
import pandas as pd


def make_prefix(dp, template_type):
    question = dp["question"]
    if template_type == "base":
        prefix = f"""The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a math problem. After thinking, when you finally reach a conclusion, clearly state the final result within <answer> </answer> tags. For example, <answer> 124 </answer>.\n\nUser:{question}\nAssistant: <think>"""
    elif template_type == "qwen-instruct":
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a math problem. After thinking, when you finally reach a conclusion, clearly state the final result within <answer> </answer> tags. i.e., <answer> 124 </answer>.\n<|im_end|>\n<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
    return prefix


def sample_questions(digit, n_total):
    nums_1 = random.sample(range(10 ** (digit - 1), 10**digit - 1), n_total)
    nums_2 = random.sample(range(10 ** (digit - 1), 10**digit - 1), n_total)
    questions = [
        {
            "question": f"Calculate the result of {num_1} + {num_2}.",
            "solution": str(num_1 + num_2),
        }
        for num_1, num_2 in zip(nums_1, nums_2)
    ]
    df = pd.DataFrame.from_records(questions)
    return Dataset.from_pandas(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/integer_add/instruct/5_5")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--digits", type=int, default=5)
    parser.add_argument("--train_size", type=int, default=900)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--template_type", type=str, default="qwen-instruct")

    args = parser.parse_args()

    data_source = "integer_add"
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    # Load custom JSONL dataset
    dataset = sample_questions(args.digits, TRAIN_SIZE + TEST_SIZE)
    print(len(dataset))
    print(dataset[:5])

    assert len(dataset) >= TRAIN_SIZE + TEST_SIZE
    train_dataset = dataset.select(range(TRAIN_SIZE))
    test_dataset = dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "solution": example["solution"],
            }
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "logic",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    print(train_dataset[0])
    print(test_dataset[0])

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create local directory if not exists
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

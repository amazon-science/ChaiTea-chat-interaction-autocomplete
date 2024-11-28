# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re

import pandas as pd
from datasets import load_dataset
from langdetect import detect_langs
from tqdm.notebook import tqdm

tqdm.pandas()

trunc = 250


def get_prompt(conversation):
    history = ""
    for message in conversation[:-1]:
        role = "prompter" if message["from"] == "human" else "assistant"
        text = message["value"]
        history += f"<|{role}|> {text}\n"
    return history


def build_user_prompts(dataset):
    print("building user prompts")
    user_prompts = []
    for index, row in dataset.iterrows():
        # if index % 1000 == 0:
        #     print(f"{index}/{len(dataset)}")
        conversation = row["conversations"]
        for i, message in enumerate(conversation):
            if message["from"] == "human":
                history = get_prompt(conversation[: i + 1])
                user_prompts.append(dict(id=row["id"], text=message["value"], history=history))
    return pd.DataFrame(user_prompts)


def build_word_prefixes(user_prompts):
    user_prompts_df_w_prefix = pd.DataFrame(
        [
            {
                **d,
                "prefix": d["history"] + "<|prompter|> " + " ".join(d["text"].split()[:i]),
                "prefix_k": i,
                "gt_completion": re.search(r"^.*?[.!?\n]+|.*$", " ".join(d["text"].split()[i:]), flags=re.DOTALL)
                .group()
                .strip(),
                # gt is only until end-of-line
            }
            for _, d in user_prompts.iterrows()
            for i in range(len(d["text"].split()))
        ]
    )

    return user_prompts_df_w_prefix


def build_char_prefixes(user_prompts):
    user_prompts_df_w_prefix = pd.DataFrame(
        [
            {
                **d,
                "prefix": d["history"] + "<|prompter|> " + d["text"][:i],
                "prefix_k": i,
                "gt_completion": re.search(r"^.*?[.!?\n]+|.*$", d["text"][i:], flags=re.DOTALL).group(),
            }
            for _, d in user_prompts.iterrows()
            for i in range((len(d["text"]) - trunc + d["first_eos"] if len(d["text"]) > trunc else 0), len(d["text"]))
        ]
    )
    return user_prompts_df_w_prefix


def build_prefixes(user_prompts, char=False):
    if char:
        user_prompts_df_w_prefix = build_char_prefixes(user_prompts)
    else:
        user_prompts_df_w_prefix = build_word_prefixes(user_prompts)

    # remove rows where there is no history and prefix is empty
    user_prompts_df_w_prefix = user_prompts_df_w_prefix[
        user_prompts_df_w_prefix["prefix"] != "<|prompter|> "
    ].reset_index(drop=True)

    return user_prompts_df_w_prefix


def prepare_sharegpt(save_path, char, train=False, hp_subset=None):
    filtered_dataset_path = "./datasets/sharegpt/dataset_eng_all.csv"

    # load dataset
    dataset = load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered", data_files="ShareGPT_V3_unfiltered_cleaned_split.json"
    )
    dataset = dataset.filter(lambda x: x["conversations"])
    dataset = dataset.filter(lambda x: "human" in [c["from"] for c in x["conversations"]])["train"]

    dataset = pd.DataFrame(dataset)

    if not os.path.isfile(filtered_dataset_path):
        # filter non eng conversations
        dataset["is_conv_eng"] = dataset["conversations"].apply(is_conv_eng)
        print(f"total convs: {len(dataset)}")
        filtered_dataset = dataset[dataset["is_conv_eng"]]
        print(f"english convs: {len(dataset)}")
        filtered_dataset.to_csv(filtered_dataset_path, index=False)
    else:
        filtered_dataset = pd.read_csv(filtered_dataset_path)
        filtered_dataset = dataset[dataset["id"].isin(filtered_dataset["id"].unique())].reset_index(drop=True)

    train_set = filtered_dataset.sample(frac=0.96, random_state=42)
    val_set = filtered_dataset.drop(train_set.index)

    user_prompts = build_user_prompts(val_set)
    print(f"#val user prompts: {len(user_prompts)}")

    if train:
        # train_user_prompts = build_user_prompts(train_set)
        # print(f"#train user prompts: {len(train_user_prompts)}")
        # train_user_prompts["first_eos"] = train_user_prompts["text"].apply(first_eos)
        # train_user_prompts_w_prefix = build_prefixes(train_user_prompts, char=False)
        # print(f"#train prefixes: {len(train_user_prompts_w_prefix)}")
        val_user_prompts_w_prefix = build_prefixes(user_prompts, char=False)
        print(f"#val prefixes: {len(val_user_prompts_w_prefix)}")

    user_prompts["first_eos"] = user_prompts["text"].apply(first_eos)

    sampled_user_prompts = user_prompts.sample(n=1500, random_state=42)
    if hp_subset is not None:
        sampled_user_prompts = sampled_user_prompts.sample(frac=hp_subset, random_state=42)
    user_prompts_w_prefix = build_prefixes(sampled_user_prompts, char=False)

    if not char:
        user_prompts_w_prefix = user_prompts_w_prefix[
            user_prompts_w_prefix["gt_completion"].str.startswith(" ")
            | user_prompts_w_prefix["gt_completion"].str.startswith("\n")
        ]
        user_prompts_w_prefix = user_prompts_w_prefix[user_prompts_w_prefix["gt_completion"].str.strip() != ""]

    # save to file
    user_prompts_w_prefix.to_csv(save_path, index=False)


def is_conv_eng(conv):
    # start = timeit.default_timer()
    text = "\n".join([x["value"] for x in conv])

    # Check percentage of non-English Unicode characters
    non_eng_chars = sum(1 for c in text if not c.isascii())
    total_chars = len(text)
    if non_eng_chars / total_chars > 0.05:
        return False

    lang_code = detect_language(text)
    if lang_code != "en":
        return False
    # print(timeit.default_timer() - start)
    return True


def first_eos(text):
    if len(text) < trunc:
        return 0

    res = text[-trunc:]
    # get end of sentence
    first_eos = len(re.search(r"^.*?[.!?\n]+|.*$", res, flags=re.DOTALL).group())
    return first_eos


def detect_language(text):
    try:
        detected_langs = detect_langs(text)
        lang_code = detected_langs[0].lang
    except Exception:
        lang_code = "unknown"
    return lang_code


if __name__ == "__main__":
    prepare_sharegpt(
        "./datasets/sharegpt/user_prompts_w_prefix_filtered_chars_new.csv",
        char=False,
        train=True,
    )

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

import re

import pandas as pd
from datasets import load_dataset


def get_prompt(d):
    return (d["history"] + "\n" if not pd.isna(d["history"]) and d["history"] != "" else "") + "<|prompter|> "


def count_words(text):
    if pd.isna(text):
        return 0
    return len(text.split())


def build_user_prompts(dataset, mid2parentid, mid2text, mid2role):
    prompt_ids = dataset.query("role == 'prompter'")["message_id"].tolist()
    user_prompts = []
    for mid in prompt_ids:
        history = []
        parentids = []
        parentid = mid2parentid[mid]
        cmid = mid
        while parentid is not None:
            parentids.append(parentid)

            if parentid not in mid2text:
                print(f"Error: {parentid} not in mid2text/mid2role")
                break
            history.append(f"<|{mid2role[parentid]}|> {mid2text[parentid]}")

            if cmid not in mid2parentid:
                print(f"Error: {mid} not in mid2parentid")
                break
            cmid = parentid
            parentid = mid2parentid[cmid]

        user_prompts.append(
            dict(mid=mid, text=mid2text[mid], history="\n".join(reversed(history)), parentids=list(reversed(parentids)))
        )
    return user_prompts


def build_word_prefixes(user_prompts):
    user_prompts_df_w_prefix = pd.DataFrame(
        [
            {
                **d,
                "prefix": get_prompt(d) + " ".join(d["text"].split()[:i]),
                "prefix_k": i,
                "gt_completion": re.search(r"^.*?[.!?\n]+|.*$", " ".join(d["text"].split()[i:]), flags=re.DOTALL)
                .group()
                .strip(),
                # gt is only until end-of-line
            }
            for d in user_prompts
            for i in range(len(d["text"].split()))
        ]
    )

    return user_prompts_df_w_prefix


def build_char_prefixes(user_prompts, downsample=1):
    user_prompts_df_w_prefix = pd.DataFrame(
        [
            {
                **d,
                "prefix": get_prompt(d) + d["text"][:i],
                "prefix_k": i,
                "gt_completion": re.search(r"^.*?[.!?\n]+|.*$", d["text"][i:], flags=re.DOTALL).group(),
            }
            for d in user_prompts[::downsample]
            for i in range(len(d["text"]))
        ]
    )
    return user_prompts_df_w_prefix


def build_prefixes(user_prompts, char=False, downsample=1):
    if char:
        user_prompts_df_w_prefix = build_char_prefixes(user_prompts, downsample)
    else:
        user_prompts_df_w_prefix = build_word_prefixes(user_prompts)

    # remove rows where there is no history and prefix is empty
    user_prompts_df_w_prefix = user_prompts_df_w_prefix[
        user_prompts_df_w_prefix["prefix"] != "<|prompter|> "
    ].reset_index(drop=True)

    return user_prompts_df_w_prefix


def prepare_oasst(save_path, char, hp_subset=None, split="val", downsample=1):
    # load dataset
    dataset = load_dataset("OpenAssistant/oasst2")
    dataset = {k: dataset[k].to_pandas() for k in ["train", "validation"]}
    if split == "val":
        dataset = dataset["validation"].query('lang == "en"')
    elif split == "train":
        dataset = dataset["train"].query('lang == "en"')
    mid2text = dataset[["message_id", "text"]].set_index("message_id")["text"].to_dict()
    mid2parentid = dataset[["message_id", "parent_id"]].set_index("message_id")["parent_id"].to_dict()
    mid2role = dataset[["message_id", "role"]].set_index("message_id")["role"].to_dict()

    # build prompts and prefixes for each user message
    user_prompts = build_user_prompts(dataset, mid2parentid, mid2text, mid2role)

    user_prompts_w_prefix = build_prefixes(user_prompts, char, downsample)
    if hp_subset is not None:
        sampled_mids = pd.DataFrame(user_prompts_w_prefix["mid"].unique()).sample(frac=hp_subset, random_state=42)
        user_prompts_w_prefix = user_prompts_w_prefix[user_prompts_w_prefix["mid"].isin(sampled_mids[0])]

    # save to file
    user_prompts_w_prefix.to_csv(save_path)


if __name__ == "__main__":
    prepare_oasst(
        "test_save_path",
        char=False,
        hp_subset=None,
    )

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

import argparse
import os
import pickle
import re
from os import listdir
from os.path import isfile, join

import pandas as pd
from tqdm import tqdm

from ranker import Ranker

config_parser = parser = argparse.ArgumentParser(description="", add_help=False)
parser.add_argument(
    "--model_id",
    default="mistralai_Mistral-7B-v0.1",
    # default="microsoft/Phi-3-mini-4k-instruct",
    help="model name as appears in huggingface or path to finetuned model",
    type=str,
)
parser.add_argument("--single_word", default=False, help="")
parser.add_argument("--partial", default=True, help="")
parser.add_argument("--char_prefixes", default=False, help="")
parser.add_argument("--on_reject_type", default="word", help="")
parser.add_argument("--dataset", default="oasst", help="")
parser.add_argument("--subset", default=None, help="")
parser.add_argument("--hparams", default=False, help="")
parser.add_argument(
    "--rank_by",
    default="log_likelihood",
    choices=[
        "neg_perplexity",
        "log_likelihood",
        "max_logprob",
        "min_logprob",
        "mean_neg_entropy_norm",
        "mean_neg_entropy",
        "max_neg_entropy_norm",
        "max_neg_entropy",
        "min_neg_entropy_norm",
        "min_neg_entropy",
        "sequence_length",
        "random",
    ],
)

parser.add_argument("--theta", default=None, help="")


def split_punc(text):
    return re.findall(r"[^ \s,.!?]+|[?!.,]", text.lower())


class Metrics:
    def __init__(self, saved_completions):
        self.saved_completions = saved_completions
        self.comp_dict = {}
        for completion in self.saved_completions:
            self.comp_dict[completion["dataset_row"]["prefix"].lower()] = completion

    def saved_typing_at_k(
        self,
        k,
        prefix,
        ground_truth,
        completions_fn,
        acceptance_fn,
        ranker,
        args,
        len_fn="chars",
        prefix_idx=None,
        char_prefixes=False,
    ):
        if len_fn == "chars":
            self.len_fn = len
        elif len_fn == "words":
            self.len_fn = lambda x: len(x.split())  # noqa: E731

        if char_prefixes:
            full_ground_truth = prefix + ground_truth
        else:
            full_ground_truth = prefix.strip() + " " + ground_truth.strip()

        ground_truth_words = split_punc(full_ground_truth)
        prefix_words = split_punc(prefix)
        len_to_complete = self.len_fn(full_ground_truth) - self.len_fn(prefix.strip())

        num_accepts, accepted_length, num_suggested, num_abstained = 0, 0, 0, 0
        (
            accepted_text_indices,
            accepted_text_strs,
            accepted_ranks,
            accepted_confidence,
        ) = [], [], [], []

        while self.len_fn(prefix) < self.len_fn(full_ground_truth) and prefix_words != ground_truth_words:
            # if next char is punc skip getting completions
            if not char_prefixes and (
                bool(re.fullmatch(r"[^\w\s]", ground_truth_words[len(prefix_words)]))
                or bool(re.fullmatch(r"[^\w\s]", prefix[-1]) and full_ground_truth[len(prefix)] != " ")
            ):
                completions, confidence = [], []
            else:
                completions, confidence = completions_fn(
                    k,
                    prefix,
                    prefix_idx,
                    ranker,
                    single_word=args["single_word"],
                    partial=args["partial"],
                    char_prefixes=args["char_prefixes"],
                )

            # abstain if not confident
            if args["theta"] is not None:
                k_new = len([c for c in confidence if c >= args["theta"]])
                completions, confidence = completions[:k_new], confidence[:k_new]
                if k_new == 0:
                    num_abstained += 1

            if len(completions) > 0:
                num_suggested += 1
                accepted_text = acceptance_fn(
                    prefix, completions, full_ground_truth, args.get("add_leading_space", False)
                )
                if accepted_text is not None and accepted_text.strip() != "":
                    num_accepts += 1
                    accepted_length += self.len_fn(accepted_text)
                    accepted_text_indices.append((len(prefix), len(prefix + accepted_text) + 1))
                    if char_prefixes:
                        prefix = prefix + accepted_text
                        accepted_text_strs.append(accepted_text)
                    else:
                        prefix = prefix.strip() + " " + accepted_text.strip()
                        accepted_text_strs.append(accepted_text)
                    rank = completions.index(accepted_text)
                    accepted_ranks.append(rank + 1)
                    accepted_confidence.append(confidence[rank])
                    prefix_words = split_punc(prefix)
                    continue

            # if no completions were suggested, or none of the completions is accepted, simulate the user typing another word or char
            prefix_words = split_punc(prefix)
            if args["on_reject_type"] == "char":
                prefix += full_ground_truth[len(prefix)]
            else:
                if prefix_words == ground_truth_words:
                    break
                if full_ground_truth[len(prefix)] == " ":
                    prefix = prefix.strip() + " " + ground_truth_words[len(prefix_words)]
                else:
                    prefix = prefix + ground_truth_words[len(prefix_words)]

            prefix_words = split_punc(prefix)

        min_num_clicks = len_to_complete - 1 if (len_to_complete - 1) > 0 else 1
        accepted_length_words = sum([len(t.split()) for t in accepted_text_strs])

        res = {
            f"saved@{k}": (accepted_length - num_accepts) / min_num_clicks,
            f"num_accepts@{k}": num_accepts,
            f"accepted_length_chars@{k}": accepted_length,
            f"accepted_length_words@{k}": accepted_length_words,
            f"accepted_text_indices@{k}": accepted_text_indices,
            f"accepted_text_strs@{k}": accepted_text_strs,
            f"acceptance_rate@{k}": num_accepts / num_suggested if num_suggested > 0 else 0,
            f"completed_length@{k}": accepted_length / len_to_complete,
            f"chars_per_accept@{k}": accepted_length / num_accepts if num_accepts > 0 else 0,
            f"words_per_accept@{k}": accepted_length_words / num_accepts if num_accepts > 0 else 0,
            f"accepted_ranks@{k}": accepted_ranks,
            f"mean_accepted_ranks@{k}": sum(accepted_ranks) / len(accepted_ranks) if len(accepted_ranks) > 0 else 0,
            f"mean_accepted_confidence@{k}": sum(accepted_confidence) / len(accepted_confidence)
            if len(accepted_confidence) > 0
            else 0,
            f"accepted_confidence@{k}": accepted_confidence,
            f"num_abstained@{k}": num_abstained,
            f"num_suggested@{k}": num_suggested,
            f"abstain_ratio@{k}": num_abstained / (num_abstained + num_suggested)
            if (num_abstained + num_suggested) > 0
            else 0,
        }
        return res

    def exact_match_acceptance(self, prefix, completions, full_ground_truth, add_leading_space=False):
        completions = sorted([c for c in completions if c.strip() != ""], key=len, reverse=True)
        for completion in completions:
            if add_leading_space:
                suggested_text = prefix + " " + completion
            else:
                suggested_text = prefix + completion
            if len(suggested_text) > len(full_ground_truth):
                continue

            suggested_text_words = split_punc(suggested_text)
            ground_truth_words = split_punc(full_ground_truth)
            if (
                suggested_text_words == ground_truth_words[: len(suggested_text_words)]
                and suggested_text.lower() == full_ground_truth[: len(suggested_text)].lower()
            ):
                return completion

        return None

    def get_saved_completions(
        self, k, prefix, prefix_idx, ranker, partial=True, single_word=False, char_prefixes=False
    ):
        try:
            saved_completion = self.comp_dict[prefix.lower()]
            key = saved_completion["completion"]["generated_text"]
        except Exception:
            print(f"missing completion, idx {prefix_idx}")
            return [], []
        completions = {
            key: {
                "tokens": [t["text"] for t in saved_completion["completion"]["details"]["tokens"]],
                "logprobs": [t["logprob"] for t in saved_completion["completion"]["details"]["tokens"]],
                "top_tokens_logprobs": [
                    [t1["logprob"] for t1 in t2] for t2 in saved_completion["completion"]["details"]["top_tokens"]
                ]
                if "top_tokens" in saved_completion["completion"]["details"].keys()
                else [],
            }
        }
        if "best_of_sequences" in saved_completion["completion"]["details"].keys():
            for c in saved_completion["completion"]["details"]["best_of_sequences"]:
                completions[c["generated_text"]] = {
                    "tokens": [t["text"] for t in c["tokens"]],
                    "logprobs": [t["logprob"] for t in c["tokens"]],
                    "top_tokens_logprobs": [[t1["logprob"] for t1 in t2] for t2 in c["top_tokens"]]
                    if "top_tokens" in saved_completion["completion"]["details"].keys()
                    else [],
                }
        if "" in completions.keys():
            del completions[""]

        # include all partial completions
        if partial:
            partial_completions = {}
            for _, tokens_dict in completions.items():
                for j in range(1, len(tokens_dict["tokens"])):
                    # if the next token is a new word or is the end of the sequence
                    if tokens_dict["tokens"][j].startswith(" ") or bool(
                        re.fullmatch(r"[^\w\s]", tokens_dict["tokens"][j])
                    ):
                        partial_completions["".join(tokens_dict["tokens"][:j])] = {
                            "tokens": tokens_dict["tokens"][:j],
                            "logprobs": tokens_dict["logprobs"][:j],
                            "top_tokens_logprobs": tokens_dict["top_tokens_logprobs"][:j],
                        }
            completions.update(partial_completions)

        if single_word:
            single_word_completions = {}
            for _, tokens_dict in completions.items():
                for j in range(1, len(tokens_dict["tokens"])):
                    # if the next token is a new word or is the end of the sequence
                    if tokens_dict["tokens"][j].startswith(" ") or bool(
                        re.fullmatch(r"[^\w\s]", tokens_dict["tokens"][j])
                    ):
                        single_word_completions["".join(tokens_dict["tokens"][:j])] = {
                            "tokens": tokens_dict["tokens"][:j],
                            "logprobs": tokens_dict["logprobs"][:j],
                            "top_tokens_logprobs": tokens_dict["top_tokens_logprobs"][:j],
                        }
                        break
                if len("".join(tokens_dict["tokens"]).split()) == 1:
                    single_word_completions["".join(tokens_dict["tokens"])] = {
                        "tokens": tokens_dict["tokens"],
                        "logprobs": tokens_dict["logprobs"],
                        "top_tokens_logprobs": tokens_dict["top_tokens_logprobs"],
                    }

            completions = single_word_completions

        # sort candidates
        sorted_completions, confidence = ranker.rank(prefix, completions)

        # return top k completions and confidence
        return sorted_completions[:k], confidence[:k]


models_list = [
    "microsoft_Phi-3.5-mini-instruct",
    "mistralai/Mistral-7B-v0.1",
    "HuggingFaceH4/zephyr-7b-beta",
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "google/gemma-7b",
    "google/gemma-7b-it",
    "microsoft/Phi-3-mini-4k-instruct",
    "openai-community/gpt2-xl",
    "HuggingFaceTB/SmolLM-1.7B",
    "state-spaces/mamba-2.8b",
    "xiuyul/mamba-2.8b-zephyr",
]


def get_src_out_dirs(args):
    src_dir = f"./completions/{args['dataset']}"
    out_dir = f"./results/saved_at_k/{args['dataset']}"
    if args["single_word"]:
        out_dir += "/baselines/single_word"
    if args["char_prefixes"]:
        if args["dataset"] == "oasst":
            src_dir += "/char/full_149K"
            out_dir += "/char/full_149K"
        elif args["dataset"] == "sharegpt":
            src_dir += "/char/sample_144K"
            out_dir += "/char/sample_144K"
        if args["on_reject_type"] == "word":
            out_dir += "_on_reject_type_word"
    else:
        if args["dataset"] == "oasst":
            src_dir += "/word/val"
            out_dir += "/word/val"
        elif args["dataset"] == "sharegpt":
            src_dir += "/word/sample_22K"
            out_dir += "/word/sample_22K"
        elif "prism" in args["dataset"]:
            src_dir += "/word/val"
            out_dir += "/word/val"

    if args["subset"] is not None:
        src_dir += f"/subset_{args['subset']}"
        out_dir += f"/subset_{args['subset']}"

    if args["hparams"]:
        src_dir = "./completions/oasst/word_subset_0.25"
        out_dir = "./results/saved_at_k/oasst_subset_0.25"
    return src_dir, out_dir


def get_k_vals(args):
    k_vals = [1, 2, 3, 5, 10, 20, 50, 100]
    if args["single_word"]:
        k_vals = [1, 2, 3, 4, 5]
    if args["hparams"]:
        k_vals = [100]
    return k_vals


if __name__ == "__main__":
    args = vars(parser.parse_args())
    args["theta"] = float(args["theta"]) if args["theta"] is not None else None
    print(args)
    src_dir, out_dir = get_src_out_dirs(args)
    k_vals = get_k_vals(args)
    if args["model_id"] is not None:
        models_list = [args["model_id"]]

    for model in models_list:
        if "mamba" in model and "entropy" in args["rank_by"]:
            continue
        model = model.replace("/", "_")
        model_dir = join(src_dir, model)

        files = sorted(
            [
                f
                for f in listdir(model_dir)
                if isfile(join(model_dir, f)) and ("best_of:5" in f)
                # if isfile(join(model_dir, f)) and ("best_of:5" in f and "max_new_tokens:20" in f)
                # and (("best_of:5" in f and "max_new_tokens:20" in f) or ("best_of:1" in f and "max_new_tokens:5" in f))
            ]
        )
        if args["single_word"] or not args["partial"]:
            files = sorted(
                [
                    f
                    for f in listdir(model_dir)
                    if isfile(join(model_dir, f)) and ("best_of:5" in f and "max_new_tokens:20" in f)
                ]
            )

        if args["hparams"]:
            files = sorted([f for f in listdir(model_dir) if isfile(join(model_dir, f))])

        for f in files:
            # if "history_len:None" not in f:
            #     continue
            output_path = f"{out_dir}/{model}/{f.replace('.pkl','')}__rank_by:{args['rank_by']}.csv"
            print(f"results output file: {output_path}")
            if not isfile(output_path):
                print(f"Running metric results for model {model}, config: {f}")
                rows = []
                model_completions = []
                with open(join(model_dir, f), "rb") as cf:
                    model_completions = pickle.load(cf)

                print(f"found model completions, length: {len(model_completions)}")
                # if len(model_completions) < 6795:
                #     continue
                assert len(model_completions) > 0
                metrics = Metrics(saved_completions=model_completions)
                ranker = Ranker(
                    saved_completions=model_completions,
                    confidence_measure=args["rank_by"],
                    model_name=args["model_id"],
                )

                for i, c in enumerate(tqdm(model_completions)):
                    # if args["char_prefixes"] and i % 10 != 0:
                    #     continue

                    rows.append(c["dataset_row"].copy())
                    for k in k_vals:
                        prefix = c["dataset_row"]["prefix"]
                        ground_truth = c["dataset_row"]["gt_completion"]

                        if (
                            args["on_reject_type"] == "word"
                            and args["char_prefixes"]
                            and not ground_truth.startswith(" ")
                            and not ground_truth.startswith("\n")
                        ):
                            continue

                        saved_typing_at_k_results = metrics.saved_typing_at_k(
                            k,
                            prefix,
                            ground_truth,
                            completions_fn=metrics.get_saved_completions,
                            acceptance_fn=metrics.exact_match_acceptance,
                            ranker=ranker,
                            args=args,
                            prefix_idx=i,
                            char_prefixes=args["char_prefixes"],
                        )
                        rows[-1].update(saved_typing_at_k_results)

                results_df = pd.DataFrame(rows)
                os.makedirs(join(out_dir, model), exist_ok=True)
                results_df.to_csv(output_path, index=False)

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

import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Ranker:
    def __init__(self, saved_completions, confidence_measure, model_name=None):
        self.saved_completions = saved_completions
        self.confidence_measure = confidence_measure
        self.rank_fn = getattr(self, self.confidence_measure)
        random.seed(42)

    def init_model(self, model_name, gpu_id=3):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, truncation_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map=f"cuda:{gpu_id}",
            # device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        self.save_counter = 0

    def rank(self, prefix, completions):
        completions = self.rank_fn(prefix, completions)
        return self.sort_by_key(completions, self.confidence_measure)

    def sort_by_key(self, completions, key):
        sorted_completions = dict(
            sorted(
                completions.items(),
                key=lambda item: item[1][key],
                reverse=True,
            )
        )
        return list(sorted_completions.keys()), [c[key] for c in sorted_completions.values()]

    def neg_entropy(self, tokens_logprobs, normalize):
        max_len = max(len(arr) for arr in tokens_logprobs)
        probs = np.exp(np.stack([np.pad(arr, (0, max_len - len(arr)), constant_values=0) for arr in tokens_logprobs]))
        if normalize:
            probs = probs / np.sum(probs, axis=1, keepdims=True)
        neg_entropy = np.sum(probs * np.log(probs), axis=1)
        return neg_entropy

    def neg_perplexity(self, prefix, completions):
        for _, tokens_dict in completions.items():
            tokens_dict["neg_perplexity"] = np.exp(np.mean(tokens_dict["logprobs"]))
        return completions

    def log_likelihood(self, prefix, completions):
        for _, tokens_dict in completions.items():
            tokens_dict["log_likelihood"] = np.sum(tokens_dict["logprobs"])
        return completions

    def max_logprob(self, prefix, completions):
        for _, tokens_dict in completions.items():
            tokens_dict["max_logprob"] = np.max(tokens_dict["logprobs"])
        return completions

    def min_logprob(self, prefix, completions):
        for _, tokens_dict in completions.items():
            tokens_dict["min_logprob"] = np.min(tokens_dict["logprobs"])
        return completions

    def mean_neg_entropy_norm(self, prefix, completions):
        for _, tokens_dict in completions.items():
            tokens_dict["mean_neg_entropy_norm"] = np.mean(
                self.neg_entropy(tokens_dict["top_tokens_logprobs"], normalize=True)
            )
        return completions

    def mean_neg_entropy(self, prefix, completions):
        for _, tokens_dict in completions.items():
            tokens_dict["mean_neg_entropy"] = np.mean(
                self.neg_entropy(tokens_dict["top_tokens_logprobs"], normalize=False)
            )
        return completions

    def max_neg_entropy_norm(self, prefix, completions):
        for _, tokens_dict in completions.items():
            tokens_dict["max_neg_entropy_norm"] = np.max(
                self.neg_entropy(tokens_dict["top_tokens_logprobs"], normalize=True)
            )
        return completions

    def min_neg_entropy_norm(self, prefix, completions):
        for _, tokens_dict in completions.items():
            tokens_dict["min_neg_entropy_norm"] = np.min(
                self.neg_entropy(tokens_dict["top_tokens_logprobs"], normalize=True)
            )
        return completions

    def max_neg_entropy(self, prefix, completions):
        for _, tokens_dict in completions.items():
            tokens_dict["max_neg_entropy"] = np.max(
                self.neg_entropy(tokens_dict["top_tokens_logprobs"], normalize=False)
            )
        return completions

    def min_neg_entropy(self, prefix, completions):
        for _, tokens_dict in completions.items():
            tokens_dict["min_neg_entropy"] = np.min(
                self.neg_entropy(tokens_dict["top_tokens_logprobs"], normalize=False)
            )
        return completions

    # TODO: deal with single token output
    def mean_t1_t2_ratio(self, prefix, completions):
        for _, tokens_dict in completions.items():
            probs = np.exp(np.stack(tokens_dict["top_tokens_logprobs"]))
            t1 = probs[:, 0]
            t2 = probs[:, 1]
            tokens_dict["mean_t1_t2_ratio"] = (t1 / t2).mean()
        return completions

    def mean_t1_t2_gap(self, prefix, completions):
        for _, tokens_dict in completions.items():
            probs = np.exp(np.stack(tokens_dict["top_tokens_logprobs"]))
            t1 = probs[:, 0]
            t2 = probs[:, 1]
            tokens_dict["mean_t1_t2_gap"] = (t1 - t2).mean()
        return completions

    def sequence_length(self, prefix, completions):
        for c, tokens_dict in completions.items():
            tokens_dict["sequence_length"] = -len(c)
        return completions

    def random(self, prefix, completions):
        for _, tokens_dict in completions.items():
            tokens_dict["random"] = random.random()
        return completions

    def oracle_ranker(self, completions, prefix, ground_truth, metrics):
        accepted_text = metrics.exact_match_acceptance(prefix, completions, prefix.strip() + " " + ground_truth)
        if accepted_text is None:
            return [], []
        return [accepted_text], [1.0]

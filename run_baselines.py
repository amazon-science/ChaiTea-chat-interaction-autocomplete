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
import subprocess
import time

import docker
import pandas as pd

from src.inference import TGIRunner
from src.prepare_oasst import prepare_oasst
from src.prepare_sharegpt import prepare_sharegpt

# TGI_IMAGE = "ghcr.io/huggingface/text-generation-inference:2.0.2"
TGI_IMAGE = "ghcr.io/huggingface/text-generation-inference:latest"
VOLUME = "tmp/runtime/{model}/"
LOCAL_ROOT_PATH = os.path.split(os.path.abspath(__file__))[0]
S3_ROOT_PATH = "s3://<bucket_name>/<prefix>/"
hf_token_path = os.path.join(LOCAL_ROOT_PATH, "resources/hf_token.txt")


config_parser = parser = argparse.ArgumentParser(description="inference args", add_help=False)
parser.add_argument(
    "--model_id",
    default="mistralai/Mistral-7B-v0.1",
    help="model name as appears in huggingface or path to finetuned model",
)
parser.add_argument("--gpu_id", default=2, type=int, help="gpu id")
parser.add_argument("--best_of", default=5, type=int, help="number of generated output sequences")
parser.add_argument("--max_new_tokens", default=20, type=int, help="save logprob data for top n tokens")
parser.add_argument("--top_n_tokens", default=5, type=int, help="save logprob data for top n tokens")
parser.add_argument("--temperature", default=1.0, type=float, help="")
parser.add_argument("--top_p", default=1.0, type=float, help="")
parser.add_argument("--prefixes", default="word", type=str, help="complete every char or word")
parser.add_argument("--dataset", default="sharegpt", type=str, help="dataset to run inference on")
parser.add_argument(
    "--hp_subset", default=None, type=float, help="fraction of dataset to sample for hparam exploration"
)
parser.add_argument("--history_len", default=None, type=int, help="the amount of characters to include in the prompt")


def prepare_model(s3_model_path, model_name):
    local_model_path = os.path.join(LOCAL_ROOT_PATH, f"tmp/runtime/{model_name}/model_files/")

    os.makedirs(local_model_path, mode=0o750, exist_ok=True)

    subprocess.run(
        ["aws", "s3", "sync", s3_model_path, local_model_path],
        capture_output=True,
        text=True,
    )

    return local_model_path


if __name__ == "__main__":
    args = vars(parser.parse_args())
    print(args)

    # prepare dataset
    dataset_path = os.path.join(LOCAL_ROOT_PATH, f"datasets/{args["dataset"]}/")
    dataset_save_name = f"{args["dataset"]}/{args["prefixes"]}"
    if args["hp_subset"] is not None:
        dataset_save_name += f"/subset_{args["hp_subset"]}"
        dataset_path += f"subset_{args["hp_subset"]}"

    if args["dataset"] == "oasst":
        dataset_prepare_fn = prepare_oasst
        if args["prefixes"] == "word":
            dataset_path += "word_user_prompts_w_prefix.csv"
        elif args["prefixes"] == "char":
            dataset_path += "char_user_prompts_w_prefix_full_149K.csv"
            dataset_save_name += "/full_149K"

    elif args["dataset"] == "sharegpt":
        dataset_prepare_fn = prepare_sharegpt
        if args["prefixes"] == "word":
            dataset_path += "word_user_prompts_w_prefix_sample_22K.csv"
            dataset_save_name += "/sample_22K"
        elif args["prefixes"] == "char":
            dataset_path += "char_user_prompts_w_prefix_sample_144K.csv"
            dataset_save_name += "/sample_144K"

    if not os.path.isfile(dataset_path):
        dataset_prepare_fn(save_path=dataset_path, char=args["prefixes"] == "char", hp_subset=args["hp_subset"])

    dataset = pd.read_csv(dataset_path, index_col=0)

    model_name = args["model_id"]
    if "finetuned" in args["model_id"]:
        s3_model_path = "s3://<bucket_name>/<prefix_to_model_files>/"
        model_local_path = prepare_model(s3_model_path, model_name)
        model = "/data/model_files"
    else:
        model = model_name
        model_local_path = None
    model_save_name = model_name.replace("/", "_")

    completions_save_path = (
        LOCAL_ROOT_PATH
        + f"/completions/{dataset_save_name}/{model_save_name}/"
        + "__".join([f"{k}:{v}" for k, v in args.items() if k not in ["model_id", "gpu_id"]])
        + ".pkl"
    )

    print(f"completions save path: {completions_save_path}")

    # if output already exists, skip all inference steps
    try:
        completions = pickle.load(open(completions_save_path, "rb"))
    except Exception:
        completions = []

    if len(completions) < len(dataset):
        # start the TGI container
        tgi_run = TGIRunner(
            model,
            args,
            TGI_IMAGE,
            os.path.join(LOCAL_ROOT_PATH, VOLUME.format(model=model_name)),
            hf_token=open(hf_token_path, "r").read(),
            gpu_id=args["gpu_id"],
        )
        print("container up successfully for", model_name)

        # run dataset
        print(f"running inference on dataset {args['dataset']}")
        tgi_run.run_dataset(
            dataset,
            save_path=completions_save_path,
            local_model_path=model_local_path,
            existing_completions=completions,
        )

        # stop the container
        client = docker.from_env()
        if tgi_run.container in client.containers.list():
            tgi_run.container.kill()
            time.sleep(30)  # wait for container to die
        print("container down successfully for", model_name)

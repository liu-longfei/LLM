import os
# 先把离线环境变量放最前面
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_LOCAL_FILES_ONLY"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_datasets_cache"
os.environ["HF_MODULES_CACHE"] = "/tmp/hf_modules_cache"
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors

# Swap this import per-policy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# load a policy
model_id = "/home/user/ubuntu/pretrain/smolvla_base"  # <- swap checkpoint
vlm_id = "/home/user/ubuntu/pretrain/SmolVLM2-500M-Video-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = SmolVLAConfig.from_pretrained(model_id)
config.device = str(device)
config.vlm_model_name = vlm_id
policy = SmolVLAPolicy.from_pretrained(model_id, config=config, strict=True).to(device).eval()

preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_id,
    preprocessor_overrides={
        "device_processor": {"device": str(device)},
        "rename_observations_processor": {
            "rename_map": {
                "observation.images.up": "observation.images.camera1",
                "observation.images.side": "observation.images.camera2",
            }
        },
        "tokenizer_processor": {"tokenizer_name": vlm_id},
    },
)
# load a lerobotdataset
# dataset = LeRobotDataset("lerobot/libero")
dataset = LeRobotDataset(
    "local/svla_so101_pickplace",
    root="/home/user/ubuntu/datasets/svla_so101_pickplace",
)

# pick an episode
episode_index = 0

# each episode corresponds to a contiguous range of frame indices
from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
to_idx   = dataset.meta.episodes["dataset_to_index"][episode_index]

# get a single frame from that episode (e.g. the first frame)
frame_index = from_idx
frame = dict(dataset[frame_index])

batch = preprocess(frame)
with torch.inference_mode():
    pred_action = policy.select_action(batch)
    # use your policy postprocess, this post process the action
    # for instance unnormalize the actions, detokenize it etc..
    pred_action = postprocess(pred_action)
    print("pred_action:", pred_action)
    print("pred_action shape:", pred_action.shape)

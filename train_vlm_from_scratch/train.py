from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
import zipfile
import io
import os
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from typing import List, Dict, Any


class VLMConfig(PretrainedConfig):
    model_type = "vlm_model"
    def __init__(self, llm_model_path = '/home/user/ubuntu/pretrain/Qwen2.5-0.5B-Instruct',
                 vision_model_path = '/home/user/ubuntu/pretrain/siglip-so400m-patch14-384',
                 freeze_vision_model = True,
                 image_pad_num = 49,
                 **kwargs):
        self.vision_model_path = vision_model_path
        self.llm_model_path = llm_model_path
        self.freeze_vision_model = freeze_vision_model
        self.image_pad_num = image_pad_num
        super().__init__(**kwargs)


class VLM(PreTrainedModel):
    config_class = VLMConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vision_model = AutoModel.from_pretrained(self.config.vision_model_path)
        self.processor = AutoProcessor.from_pretrained(self.config.vision_model_path)
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_path)
        self.linear1 = nn.Linear(self.vision_model.config.vision_config.hidden_size*4, self.llm_model.config.hidden_size)
        self.linear2 = nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size)
        if self.config.freeze_vision_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        for param in self.llm_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, labels, pixel_values, attention_mask=None):
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)
        

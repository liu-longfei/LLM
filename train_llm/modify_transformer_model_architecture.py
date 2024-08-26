import os
os.environ["CUDA_VISIBLE_DEVICES"] = '6'

import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.models.qwen2 import modeling_qwen2

class Qwen2MLP_V1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.cus_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        print('Qwen2MLP_V1')
        # return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))
        return self.cus_proj(self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state)))


modeling_qwen2.Qwen2MLP = Qwen2MLP_V1


# model_path = '/model/base_model/qwen/Qwen1.5-0.5B-Chat'
model_path = '/home/W0016778/code/dl/sft/MedicalLLM/qwen_medical3/checkpoint-10000'

from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

# Now you do not need to add "trust_remote_code=True"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
print(model)

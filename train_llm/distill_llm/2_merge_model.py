from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "/home/W0016778/pretrain/Qwen2.5-0.5B-Instruct"
checkpoint_path = "./results/checkpoint-339"
save_merge_model = "./final_model/20240220_1503"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

p_model = PeftModel.from_pretrained(model, model_id=checkpoint_path)

merge_model = p_model.merge_and_unload()
merge_model.save_pretrained(save_merge_model)
tokenizer.save_pretrained(save_merge_model)

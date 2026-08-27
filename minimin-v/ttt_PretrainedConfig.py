from transformers import PretrainedConfig

class MyModelConfig(PretrainedConfig):
    model_type = "my_custom_model"  # 自定义标识

cfg = MyModelConfig(hidden_size=256)
# 保存配置文件
cfg.save_pretrained("./my_config")

new_cfg = MyModelConfig.from_pretrained("./my_config")
print(new_cfg.model_type)  
# 输出：my_custom_model

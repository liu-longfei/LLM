import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载已保存的模型和分词器
# model = GPT2LMHeadModel.from_pretrained('/model/base_model/gpt2')
# tokenizer = GPT2Tokenizer.from_pretrained('/model/data/datasets/code-search-net-tokenizer')
model = GPT2LMHeadModel.from_pretrained('/home/W0016778/code/dl/pretrain/save_model')
tokenizer = GPT2Tokenizer.from_pretrained('/model/data/datasets/code-search-net-tokenizer')

# 输入文本
input_text = "你是谁"

# 将输入文本编码成tokens
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# 生成输出
output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50)

# 解码输出tokens
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print('----------------------------------')
print(output_text)

from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/model/qwen/Qwen-72B-Chat", trust_remote_code=True)

input_text = ""

# print(len(input_text))
tokens = tokenizer.encode(input_text, add_special_tokens=True)

# print(type(tokens))
# print(tokens[:1000])

print(f"传递给模型的 token 数: {len(tokens)}")


import requests, json

def chat_qwen_72B(messages):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-ChtJNYJD1sm5FqwA7bE8EfFa3eE847Fa9758E5626d64Cc9a"
    }
    data = {
        "model": "Qwen-72B-Chat",
        "messages": messages,
        "top_p": 0.1,
        "temperature": 0.2
    }

    response = requests.post("http://70.182.56.14:11000/v1/chat/completions", data=json.dumps(data), headers=headers).content
    return str(response, encoding='utf-8')


messages = [{"role": "system", "content": input_text}]
print(chat_qwen_72B(messages))

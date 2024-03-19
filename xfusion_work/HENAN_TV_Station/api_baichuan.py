from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
import uvicorn, json, datetime
import torch


app = FastAPI()


@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')

    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    model.generation_config.top_p = top_p
    model.generation_config.temperature = temperature

    messages = []
    # messages.append({"role": "user", "content": f"{prompt}"})
    messages.append({"role": "system", "content": f"{prompt}"})
    response = model.chat(tokenizer,
                            messages)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    return answer


if __name__ == '__main__':
    # model_path = '/home/data/transformers/baichuan-inc/Baichuan2-7B-Chat'  # baichuan2-7B
    # model_path = '/home/data/transformers/baichuan-inc/Baichuan2-13B-Chat'  # baichuan2-13B
    model_path = '/home/z00013696/Models/sql_sft_11_24'  # 微调
    # model_path = ''
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=9600, workers=1)

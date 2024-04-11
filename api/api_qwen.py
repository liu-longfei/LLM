import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig
import uvicorn, json, datetime


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

    response = model.chat(tokenizer,
                            prompt,
                            history=None)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "status": 200,
        "time": time
    }
    # print('-----------------------------')
    # print(response)
    # print('=============================')
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    return answer


if __name__ == '__main__':
    # model_path = '/home/data/transformers/baichuan-inc/Baichuan2-7B-Chat'  # baichuan2-7B
    # model_path = '/home/data/transformers/baichuan-inc/Baichuan2-13B-Chat'  # baichuan2-13B
    model_path = '/model/base_model/qwen/Qwen-7B-Chat' 
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()
    model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)

    uvicorn.run(app, host='0.0.0.0', port=9600, workers=1)
    

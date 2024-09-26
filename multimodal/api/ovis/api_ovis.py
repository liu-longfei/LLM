import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import io
import torch
from PIL import Image
from fastapi import FastAPI, Request, File, UploadFile, Form
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig
import uvicorn, json, datetime


app = FastAPI()


@app.post("/")
async def create_item(query: str = Form(...), file: UploadFile = File(...)):
    global model, text_tokenizer, visual_tokenizer
    try:
        # 获取图片和问题
        query = f'<image>\n{query}'
        image = await file.read()
        image = io.BytesIO(image)
        image = Image.open(image)

        # format conversation
        prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image])
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
        pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]

        # generate output
        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=1024,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=model.generation_config.eos_token_id,
                pad_token_id=text_tokenizer.pad_token_id,
                use_cache=True
            )
            output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
            output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
            print(f'Output:\n{output}')
        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        answer = {
            "response": output,
            "status": 200,
            "time": time
        }
        log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(output) + '"'
        print(log)
    except Exception as e:
        print(e)
        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        answer = {
            "response": f"解析错误{e}",
            "status": 500,
            "time": time
        }
    return answer


if __name__ == '__main__':
    model_path = "/root/models/Ovis1.6-Gemma2-9B"
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.bfloat16,
                                                 multimodal_max_length=8192,
                                                 trust_remote_code=True).cuda()
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()
    uvicorn.run(app, host='0.0.0.0', port=9600, workers=1)
    

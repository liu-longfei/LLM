from fastapi import FastAPI, Request
import uvicorn, json, datetime


app = FastAPI()


@app.post("/")
async def create_item(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    query = json_post_list.get('query')
    print(query)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "query": query,
        "response": "蔡徐坤打球是人们在b站上发表的恶搞视频，可能姿势比较滑稽，球技可能也是一般。",
        "status": 200,
        "time": time
    }
    print(answer)
    return answer


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=9700, workers=1)

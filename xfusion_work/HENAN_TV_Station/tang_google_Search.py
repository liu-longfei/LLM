import requests
from fastapi import FastAPI, Request
import uvicorn, json, datetime

google_search_key = "AIzaSyCrvgMenP9WZWm_ghLy-Z0cMw5dAW5475A"
google_cx_id = "d3e7f1ca4932c412b"
base_url = "https://www.googleapis.com/customsearch/v1"


def search_with_google_api(search_key):
    url = f'{base_url}?q={search_key}&cx={google_cx_id}&key={google_search_key}&c2coff=1&start=1&end=20&dateRestrict=m[1]'
    response = requests.get(url, verify=False)
    data = response.json()
    items = data.get('items', [])
    print('----------------------------------------------------')
    print(items)
    print('----------------------------------------------------')
    result = '\n'.join([item['snippet'] for item in items])
    return result


app = FastAPI()


@app.post("/")
async def create_item(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    query = json_post_list.get('query')
    result = search_with_google_api(query)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "query": query,
        "response": result,
        "status": 200,
        "time": time
    }
    print(answer)
    return answer


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=9700, workers=1)

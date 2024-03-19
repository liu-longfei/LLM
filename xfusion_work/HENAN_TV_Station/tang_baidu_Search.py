import requests
from bs4 import BeautifulSoup

from fastapi import FastAPI, Request
import uvicorn, json, datetime


def scrape_baidu_search_results(keyword):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36'
    }
    url = f"https://www.baidu.com/s?q1={keyword}"

    response = requests.get(url, headers=headers, verify=False)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')

    index = 0
    save_num = 3  # 保留几条搜索结果
    result_str = ''
    for result in soup.select('.result'):
        if index == save_num:
            return result_str
        index +=1

        title_elem = result.select('.t > a')[0]
        link = title_elem['href']

        try:
            page_response = requests.get(link, headers=headers, verify=False)
            page_response.raise_for_status()

            page_soup = BeautifulSoup(page_response.text, 'html.parser')

            content = page_soup.get_text()

            # 处理搜索结果
            lines = (line.strip() for line in content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = "\n".join(chunk for chunk in chunks if chunk)
            result_str += content

        except requests.exceptions.RequestException as e:
            print(f"Error requesting page {link}: {e}")

    return result_str


app = FastAPI()


@app.post("/")
async def create_item(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    query = json_post_list.get('query')
    result = scrape_baidu_search_results(query)
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

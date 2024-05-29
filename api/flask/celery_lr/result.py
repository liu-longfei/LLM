import requests


# api/sample/query
url = "http://127.0.0.1:8000/"
data = {"name": "小绿"}
response = requests.get(url)

print(response.text)

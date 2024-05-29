import requests


# api/sample/query
url = "http://127.0.0.1:5000/"
data = {"name": "小明"}
response = requests.get(url, json=data)

print(response.text)

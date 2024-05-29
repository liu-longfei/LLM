from flask import Flask, jsonify, request
import asyncio
import uuid
import time


app = Flask(__name__)


@app.route("/", methods=['POST', 'GET'])
def create_template():
    name = request.json.get('name')
    print("begin", name)
    time.sleep(20)
    print("end", name)
    data = {
        "name": name
    }

    return jsonify(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

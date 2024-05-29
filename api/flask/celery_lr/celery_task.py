from flask import Flask
from celery import Celery
import time


app = Flask(__name__)

# 配置 Celery
app.config['broker_url'] = 'redis://localhost:6379/0'
app.config['result_backend'] = 'redis://localhost:6379/1'
app.config.task_retry_limit = 30

print(f'---------==============-----------{app.name}')
# 初始化 Celery
celery = Celery(app.name, broker=app.config['broker_url'], backend=app.config['result_backend'])
celery.conf.update(app.config)


@app.route('/')
def index():
    # 触发一个 Celery 异步任务
    time_1 = time.time()
    task = fun_cost_time.delay('write_db', 5)
    task2 = fun_cost_time_2.delay('write_db', 10)
    task3 = fun_cost_time_3.delay('write_db', 10)
    task4 = fun_cost_time_4.delay('write_db', 10)
    print('-----------------cost time:', time.time() - time_1)
    # return f'Task ID: {task.id}'
    # return f'Task ID: {task.id}, {task2.id}'
    # return f'Task ID: {task.id}, {task2.id}, {task3.id}'
    return f'Task ID: {task.id}, {task2.id}, {task3.id}, {task4.id}'


# 定义一个 Celery 异步任务
@celery.task
def fun_cost_time(name, sleep_time):
    print('inner fun_cost_time')
    # 任何耗时的操作
    # import pdb;pdb.set_trace()
    time.sleep(1)
    print(f'---1---{name} fun1 cost time: {sleep_time}')

    # 返回一些结果，这些结果将会存储在 Celery 的结果后端中
    return 'write_db'


@celery.task
def fun_cost_time_2(name, sleep_time):
    print('inner fun_cost_time_2')
    # 任何耗时的操作
    # import pdb;pdb.set_trace()
    time.sleep(2)
    print(f'---2---{name} fun2 cost time: {sleep_time}')

    # 返回一些结果，这些结果将会存储在 Celery 的结果后端中
    return 'write_db_2'


@celery.task
def fun_cost_time_3(name, sleep_time):
    print('inner fun_cost_time_3')
    # import pdb;pdb.set_trace()
    # 任何耗时的操作
    time.sleep(3)
    print(f'---3---{name} fun3 cost time: {sleep_time}')

    # 返回一些结果，这些结果将会存储在 Celery 的结果后端中
    return 'write_db_3'


@celery.task
def fun_cost_time_4(name, sleep_time):
    print('inner fun_cost_time_4')
    # import pdb;pdb.set_trace()
    # 任何耗时的操作
    time.sleep(4)
    print(f'---4---{name} fun4 cost time: {sleep_time}')

    # 返回一些结果，这些结果将会存储在 Celery 的结果后端中
    return 'write_db_4'


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)

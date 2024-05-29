from celery.result import AsyncResult
from celery_task import celery


async_result = AsyncResult(id="0bc8dead-e263-417a-919a-ba5155909be1", app=celery)

if async_result.successful():
    result = async_result.get()
    print(result)
    # result.forget()  # 将结果删除
elif async_result.failed():
    print("执行失败")
    print(async_result.state)
elif async_result.state == "PENDING":
    print("任务等待被执行")
elif async_result.state == "RETRY":
    print("任务失败异常，正在重试")
elif async_result.state == 'STARTED':
    print('任务已经开始被执行')
'''
Task ID: e59ba406-1642-4864-96e9-b4b315e54b2e, 4c1c3f94-f282-4e16-8067-19769250af52, 6bd7ca14-2e58-4850-b797-f8809f9919c9, 0e3a3104-aa69-45e3-85c6-ad2a6c7d5d6b
'''

## 重启wsl

```bash
#停止LxssManager服务
net stop LxssManager  

#启动LxssManager服务
net start LxssManager  
```

## 将路径加入到环境变量

```python
import os
import sys

# 获取当前文件所在的路径
current_path = os.path.dirname(os.path.abspath(__file__))

# 将当前路径添加到包的搜索路径中
sys.path.append(current_path)
```

## gpt4

非原版：

https://api.chatanywhere.com.cn
sk-5KkJyD6fPjqsixyy3VGQFPblgn44zQiAbLTjedbHrtxItiVN

原版：

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

## 命令行快捷键

* **Ctrl + U**：删除光标前的所有内容。
* **Ctrl + K**：删除光标后的所有内容。
* **Ctrl + A**：将光标移动到行首（等同于`Home`键）。
* **Ctrl + E**：将光标移动到行尾（等同于`End`键）。
* **Ctrl + L**：清屏，相当于执行 `clear` 命令。

## markdown学习

https://blog.csdn.net/qq_55376781/article/details/136438224

# lr

## 旋转位置编码

https://zhuanlan.zhihu.com/p/649756898
https://zhuanlan.zhihu.com/p/642884818
https://www.bilibili.com/video/BV1Xi421R7ev/?spm_id_from=333.337.search-card.all.click&vd_source=fffd0a55d832fc312ff82010c51836f8
https://github.com/chunhuizhang/personal_chatgpt/blob/main/tutorials/llama2_src_03_RoPE_apply_rotary_emb.ipynb

## VAE

https://github.com/chunhuizhang/diffusion_models_tutorials/blob/main/tutorials/Variational_Autoencoders.ipynb
https://skylyj.github.io/vae/#vae%E4%BB%A3%E7%A0%81%E5%AE%9E%E7%8E%B0
https://www.bilibili.com/video/BV1Gx411E7Ha/?p=35&vd_source=fffd0a55d832fc312ff82010c51836f8
https://www.bilibili.com/video/BV1QC4y1A7Gs/?spm_id_from=333.337.search-card.all.click&vd_source=fffd0a55d832fc312ff82010c51836f8

## KV cache

<img alt="img_1.png" height="50%" src="img_1.png" width="50%"/>

## BatchNorm、LayerNorm和GroupNorm

https://blog.csdn.net/qq_43426908/article/details/123119919

## re

re.escape() 函数用于将字符串中所有可能被解释为正则表达式操作符的字符进行转义（即前面加反斜杠）。这使得字符串中的特殊字符如 *, +, (, ) 被视为普通字符

## vscode debug

一般情况下，大家都是使用deepspeed、torchrun运行代码。参数都特别多，然后都是使用`sh xxxx.sh`启动脚本。

#### 在python代码里面（最前面加上这句话）

```python
import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass
```

### 下载添加信用证书

```python
import os
os.environ['REQUESTS_CA_BUNDLE'] = r"D:\data\software\Xfusion_Rootcert.crt"
```

#### 在vscode的launch.json的configuration里面，加上这个配置

```python
{
            "name": "sh_file_debug",
            "type": "debugpy",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 9501
            }
        },
```

> 注意：上面的端口号都写一样。别搞错了。

### mysql密码

1234

### 启动

1. 就正常启动，直接`sh xxx.sh`
2. 在你需要debug的python文件，打上debug断点。
3. 你看打印出来的东西，是不是出现`Waiting for debugger attach`.一般来说，都很快，就出现了。
4. 再在vscode的debug页面，选择`sh_file_debug`进行debug。
5. 就基本上完成了。确实是很方便。
6. **debug结束之后，别忘记把代码里面的 添加的代码，注销掉**

# deepspeed

https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/

chatgpt:
https://chat.leadscloud.org/

git commit --amend --no-edit

# mamba

https://blog.csdn.net/v_JULY_v/article/details/134923301

## 机器

70.182.57.2  8卡H800

XI@Xfusion@123

## ASGI和WSGI

ASGI 的全称是 Asynchronous Server Gateway Interface。它是一个 Python 的标准接口，用于在 Python web 服务器和异步 Python web 应用之间进行通信。ASGI 是 WSGI（Web Server Gateway Interface）的继承和扩展，旨在提供一个标准接口，以支持异步功能和更大的可扩展性。

ASGI 不仅支持 HTTP 通信，还支持长连接协议如 WebSocket 和其他实时功能。这使得它非常适合现代网络应用，尤其是需要高性能和实时交互的应用程序。UVicorn 是一种流行的 ASGI 服务器，它是轻量级的，使用 uvloop 和 httptools 库来实现高效的异步网络编程。

## 新建虚拟机

[申请虚拟机教程](https://wenote.welink.huaweicloud.com/wapp/pages/view/share/s/2JhHzM0PDh7L2iCf531h0Z1j0dPqRM0_oN7L2pumtD2Fq94D)

需要修改

cd /etc/sysconfig/network-scripts/

## markdown免费开源

- typedown

- marktext

- obsidian

## openai 和 embedding模型

1.openai  
OpenAI Base : http://70.182.56.16:11000/v1/  
API Key : sk-ChtJNYJD1sm5FqwA7bE8EfFa3eE847Fa9758E5626d64Cc9a  
模型名称 : Qwen2-72B-Instruct  
模型类型 : LLM  
2.embedding  
服务提供方 vllm  
baseurl : http://70.182.56.67:8000/v1  
API Key : sk-hhhhsaxijks  
模型名称 bge-zh-1.5  
模型类型 : Embedding

# python的进程、线程、协程

### 全局解释锁

全局解释器锁（Global Interpreter Lock，简称 GIL）是 Python 解释器（特别是 CPython 实现）中的一个机制，用于确保在任何时刻只有一个线程可以执行 Python 字节码。这一设计的主要目的是为了简化内存管理和避免多线程环境下的竞争条件。

#### 1. GIL 的目的

* **内存管理**：Python 使用引用计数来管理内存，确保对象的生命周期。GIL 使得在多线程环境中，引用计数的更新是线程安全的，避免了复杂的锁机制。
* **简化实现**：通过使用 GIL，Python 的实现可以避免许多多线程编程中的复杂性，使得开发者不必担心数据竞争和死锁等问题。

#### 2. GIL 的影响

* **性能瓶颈**：虽然 GIL 简化了内存管理，但它也成为了多线程程序的性能瓶颈。在 CPU 密集型任务中，多个线程无法并行执行，因为 GIL 限制了同一时间只有一个线程在执行 Python 代码。这意味着即使在多核处理器上，Python 程序的性能也可能无法得到充分利用。F
* **I/O 密集型任务**：在 I/O 密集型任务中（如网络请求、文件读写等），GIL 的影响相对较小，因为线程在等待 I/O 操作时会释放 GIL，允许其他线程运行。因此，对于 I/O 密集型应用，使用多线程仍然可以提高性能。

## python中的协程

从微观上将，协程不是真正意义上的并行。**只要协程切换的足够快，外人开起来协程就像是并行。**

##### 协程能够加快程序运行的根本原因

可以将多个协程任务的等待时间并行执行。

## docker安装postgreSQL

```yml
version: '3.8'  # 指定docker-compose文件的版本
services:
  postgresql:
    image: postgres:12.22  # 使用的镜像
    container_name: postgresql  # 容器名称
    privileged: true  # 启用特权模式
    environment:
      POSTGRES_PASSWORD: 1234  # 设置环境变量
    ports:
      - "5442:5432"  # 映射端口
    volumes:
      - ./data/postgres:/var/lib/postgresql/data  # 持久化数据
    restart: always  # 可选：设置容器重启策略
```

参考链接：[Docker-10：Docker安装PostgreSQL-阿里云开发者社区](https://developer.aliyun.com/article/1025077)

## rerank

https://www.pinecone.io/learn/series/rag/rerankers/

https://www.pinecone.io/learn/series/rag/rerankers/



# bisheng

1. 初始版本
   代码路径：D:\data\code\new_code\AI_Application_Management_Platform
   虚拟环境：new_bisheng

2. 升级langchain、langgraph、删除bisheng_langchain版本
   代码路径：D:\data\code\new_code\AI_Application_Management_Platform
   虚拟环境：bisheng_update_2

3. 同步official bisheng代码后的环境：
   代码路径：D:\data\code\new_code\AI_Application_Management_Platform
   虚拟环境：bisheng_update

# RAG优化

[一文梳理有效提升RAG效果的方法 - 大模型知识库|大模型训练|开箱即用的企业大模型应用平台|智能体开发|53AI](https://www.53ai.com/news/RAG/2024081275098.html)

[独家 | 进阶RAG-提升RAG效果-腾讯云开发者社区-腾讯云](https://cloud.tencent.com/developer/article/2432749)

https://towardsdatascience.com/advanced-rag-01-small-to-big-retrieval-172181b396d4?gi=965ad87f29ee

破解软件：

http://10.32.5.31/software/

# 提交代码

1. README.md里记录不同

2. 描述里加图片

3. 提交记录只能一个

4. 流水线要通过

5. 检查代码规范

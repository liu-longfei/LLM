# Docker Compose概述

Docker Compose是一个用于定义和运行多容器应用程序的工具。它是开启精简高效的开发和部署体验的关键。

Compose简化了对整个应用程序堆栈的控制，使您可以在一个易于理解的YAML配置文件中轻松管理服务、网络和卷。然后，只需一个命令，就可以从配置文件中创建并启动所有服务。

在所有环境中创作作品；生产、阶段、开发、测试以及CI工作流。它还具有用于管理应用程序的整个生命周期的命令：

- 启动、停止和重建服务
- 查看正在运行的服务的状态
- 流式传输正在运行的服务的日志输出
- 对服务运行一次性命令

## 介绍Compose

### 为什么用Compose？

#### Docker Compose的主要优势

使用Docker Compose提供了几个好处，可以简化容器化应用程序的开发、部署和管理：

- 简化控制：Docker Compose允许您在单个YAML文件中定义和管理多容器应用程序。这简化了编排和协调各种服务的复杂任务，使管理和复制应用程序环境变得更容易。
- 高效协作：Docker Compose配置文件易于共享，便于开发人员、运营团队和其他利益相关者之间的协作。这种协作方法可使工作流程更顺畅、问题解决速度更快，并提高整体效率。
- 快速应用程序开发：Compose缓存用于创建容器的配置。重新启动未更改的服务时，Compose会重新使用现有容器。重新使用容器意味着您可以非常快速地对环境进行更改。
- 跨环境的可移植性：Compose支持Compose文件中的变量。您可以使用这些变量为不同的环境或不同的用户自定义组合。
- 广泛的社区和支持：Docker Compose受益于一个充满活力和活跃的社区，这意味着丰富的资源、教程和支持。这个社区驱动的生态系统有助于Docker Compose的持续改进，并帮助用户有效地解决问题。

#### Docker Compose的常见用例

Compose可以以多种不同的方式使用。下面概述了一些常见的用例。

##### 开发环境

在开发软件时，在独立环境中运行应用程序并与之交互的能力至关重要。Compose命令行工具可用于创建环境并与其交互。
Compose文件提供了一种记录和配置应用程序所有服务依赖项（数据库、队列、缓存、web服务API等）的方法。使用Compose命令行工具，您可以使用单个命令为每个依赖项创建并启动一个或多个容器（docker Compose-up）。
总之，这些功能为您开始项目提供了一种方便的方式。Compose可以将多页的“开发人员入门指南”简化为一个机器可读的Compose文件和几个命令。

##### 自动化测试环境

任何连续部署或连续集成过程的一个重要部分是自动化测试套件。自动化的端到端测试需要一个运行测试的环境。Compose提供了一种为测试套件创建和销毁隔离测试环境的方便方法。通过在Compose文件中定义完整的环境，您可以通过几个命令创建和销毁这些环境：

```bash
$ docker compose up -d
$ ./run_tests
$ docker compose down
```

##### 单主机部署

Compose传统上专注于开发和测试工作流程，但随着每一次发布，我们都在更多面向生产的功能上取得了进展。
有关使用面向生产的功能的详细信息，请参见在生产中编写。

## 试用Docker Compose

本教程旨在通过指导您开发基本的Python web应用程序来介绍Docker Compose的基本概念。
该应用程序使用Flask框架，在Redis中提供了一个命中计数器，为Docker Compose在web开发场景中的应用提供了实际示例。
即使您不熟悉Python，这里演示的概念也应该是可以理解的。
这是一个非规范性的例子，只是强调了您可以使用Compose做的关键事情。

### 步骤1：设置

1.为项目创建目录：

```bash
mkdir composetest
cd composetest
```

2.在项目目录中创建一个名为app.py的文件，并将以下代码粘贴到中：

```python
import time

import redis
from flask import Flask

app = Flask(__name__)
cache = redis.Redis(host='redis', port=6379)

def get_hit_count():
    retries = 5
    while True:
        try:
            return cache.incr('hits')
        except redis.exceptions.ConnectionError as exc:
            if retries == 0:
                raise exc
            retries -= 1
            time.sleep(0.5)

@app.route('/')
def hello():
    count = get_hit_count()
    return 'Hello World! I have been seen {} times.\n'.format(count)
```

在本例中，redis是应用程序网络上redis容器的主机名，使用默认端口6379。

> 注意
>
> 请注意get_hit_count函数的编写方式。如果Redis服务不可用，此基本重试循环会多次尝试请求。这在应用程序联机时启动时很有用，但如果Redis服务在应用程序生命周期内需要随时重新启动，也会使应用程序更有弹性。在集群中，这也有助于处理节点之间的瞬时连接中断。

3.在项目目录中创建另一个名为requirements.txt的文件，并将以下代码粘贴到中：

```abc
flask
redis
```

4.创建一个Dockerfile并将以下代码粘贴到中：

```abc
# syntax=docker/dockerfile:1
FROM python:3.10-alpine
WORKDIR /code
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
RUN apk add --no-cache gcc musl-dev linux-headers
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 5000
COPY . .
CMD ["flask", "run", "--debug"]
```

了解Dockerfile
这条消息告诉Docker：

- 从Python 3.10图像开始构建一个图像。
- 将工作目录设置为/code。
- 设置flask命令使用的环境变量。
- 安装gcc和其他依赖项
- 复制requirements.txt并安装Python依赖项。
- 向映像中添加元数据，以描述容器正在侦听端口5000
- 复制当前目录。在项目中添加到workdir。在图像中。
- 将容器的默认命令设置为flask run--debug。

> 重要的
> 检查Dockerfile是否没有像.txt这样的文件扩展名。某些编辑器可能会自动附加此文件扩展名，这会在运行应用程序时导致错误。

## 步骤2：在Compose文件中定义服务

Compose简化了对整个应用程序堆栈的控制，使您可以在一个易于理解的YAML配置文件中轻松管理服务、网络和卷。
在项目目录中创建一个名为compose.yaml的文件，并粘贴以下内容：

```abc
services:
  web:
    build: .
    ports:
      - "8000:5000"
  redis:
    image: "redis:alpine"
```

这个Compose文件定义了两个服务：web和redis。
web服务使用从当前目录中的Dockerfile构建的映像。然后，它将容器和主机绑定到暴露的端口8000。此示例服务使用Flask web服务器的默认端口5000。
redis服务使用从DockerHub注册表中提取的公共redis映像。
有关compose.yaml文件的更多信息，请参阅compose的工作原理。

## 步骤3：使用Compose构建并运行您的应用程序

只需一个命令，就可以从配置文件中创建并启动所有服务。
1.从您的项目目录中，通过运行dockercompose-up来启动您的应用程序。

```abc
docker compose up

Creating network "composetest_default" with the default driver
Creating composetest_web_1 ...
Creating composetest_redis_1 ...
Creating composetest_web_1
Creating composetest_redis_1 ... done
Attaching to composetest_web_1, composetest_redis_1
web_1    |  * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)
redis_1  | 1:C 17 Aug 22:11:10.480 # oO0OoO0OoO0Oo Redis is starting oO0OoO0OoO0Oo
redis_1  | 1:C 17 Aug 22:11:10.480 # Redis version=4.0.1, bits=64, commit=00000000, modified=0, pid=1, just started
redis_1  | 1:C 17 Aug 22:11:10.480 # Warning: no config file specified, using the default config. In order to specify a config file use redis-server /path/to/redis.conf
web_1    |  * Restarting with stat
redis_1  | 1:M 17 Aug 22:11:10.483 * Running mode=standalone, port=6379.
redis_1  | 1:M 17 Aug 22:11:10.483 # WARNING: The TCP backlog setting of 511 cannot be enforced because /proc/sys/net/core/somaxconn is set to the lower value of 128.
web_1    |  * Debugger is active!
redis_1  | 1:M 17 Aug 22:11:10.483 # Server initialized
redis_1  | 1:M 17 Aug 22:11:10.483 # WARNING you have Transparent Huge Pages (THP) support enabled in your kernel. This will create latency and memory usage issues with Redis. To fix this issue run the command 'echo never > /sys/kernel/mm/transparent_hugepage/enabled' as root, and add it to your /etc/rc.local in order to retain the setting after a reboot. Redis must be restarted after THP is disabled.
web_1    |  * Debugger PIN: 330-787-903
redis_1  | 1:M 17 Aug 22:11:10.483 * Ready to accept connections
```

Compose提取Redis映像，为代码构建映像，并启动您定义的服务。在这种情况下，代码在构建时被静态复制到映像中。

2.进来http://localhost:8000/在浏览器中查看应用程序的运行情况。

如果这个问题没有解决，你也可以尝试http://127.0.0.1:8000.
您应该在浏览器中看到一条消息：

```abc
Hello World! I have been seen 1 times.
```

3.刷新页面

这个数字应该递增

```abc
Hello World! I have been seen 2 times.
```

4.切换到另一个终端窗口，然后键入docker image ls以列出本地图像。

此时列出的图像应该返回redis和web。

```abc
docker image ls

REPOSITORY        TAG           IMAGE ID      CREATED        SIZE
composetest_web   latest        e2c21aa48cc1  4 minutes ago  93.8MB
python            3.4-alpine    84e6077c7ab6  7 days ago     82.5MB
redis             alpine        9d8fa9aa0e5b  3 weeks ago    27.5MB
```

您可以使用docker inspect＜tag or id＞来检查图像。

5.停止应用程序，可以在第二个终端的项目目录中运行docker compose，也可以在启动应用程序的原始终端中点击CTRL+C。

## 步骤4：编辑Compose文件以使用Compose Watch

编辑项目目录中的compose.yaml文件以使用watch，这样您就可以预览正在运行的compose服务，这些服务在编辑和保存代码时会自动更新：

```abc
services:
  web:
    build: .
    ports:
      - "8000:5000"
    develop:
      watch:
        - action: sync
          path: .
          target: /code
  redis:
    image: "redis:alpine"
```

无论何时更改文件，Compose都会将文件同步到容器内/code下的相应位置。复制后，bundler会在不重新启动的情况下更新正在运行的应用程序。


> 注意
>
> 为了使此示例正常工作，将--debug选项添加到Dockerfile中。Flask中的--debug选项允许自动重新加载代码，从而可以在后端API上工作，而无需重新启动或重建容器。更改.py文件后，后续的API调用将使用新代码，但在这个小示例中，浏览器UI不会自动刷新。大多数前端开发服务器都包含与Compose配合使用的本机实时重载支持。

步骤5：使用Compose重新构建并运行应用程序

在项目目录中，键入docker compose watch或docker compose up --watch以构建和启动应用程序并启动文件监视模式。

```abc
docker compose watch
[+] Running 2/2
 â Container docs-redis-1 Created                                                                                                                                                                                                        0.0s
 â Container docs-web-1    Recreated                                                                                                                                                                                                      0.1s
Attaching to redis-1, web-1
         â¦¿ watch enabled
...
```

再次在web浏览器中检查Hello World消息，然后刷新以查看计数增量。

## 步骤6：更新应用程序

要查看Compose Watch的实际操作，请执行以下操作：
1.更改app.py中的问候语并保存。例如，更改Hello World！Docker给Hello的消息！：

```ab
return 'Hello from Docker! I have been seen {} times.\n'.format(count)
```

2.在浏览器中刷新应用程序。问候语应该更新，计数器应该仍在递增。

3.完成后，运行docker compose down。

## 第7步：拆分服务

使用多个Compose文件可以为不同的环境或工作流自定义Compose应用程序。这对于可能使用数十个容器、所有权分布在多个团队中的大型应用程序非常有用。
1.在项目文件夹中，创建一个名为infra.yaml的新Compose文件。
2.从compose.yaml文件中剪切Redis服务，并将其粘贴到新的infras.yaml中。请确保在文件的顶部添加服务顶级属性。您的infras.yaml文件现在应该是这样的：

```abc
services:
  redis:
    image: "redis:alpine"
```

3.在compose.yaml文件中，添加include顶级属性以及infra.yaml的路径。

```abc
include:
   - infra.yaml
services:
  web:
    build: .
    ports:
      - "8000:5000"
    develop:
      watch:
        - action: sync
          path: .
          target: /code
```

4.运行docker compose up以使用更新的compose文件构建应用程序，然后运行它。您应该会在浏览器中看到Hello world消息。

这是一个简化的例子，但它展示了include的基本原理，以及它如何使复杂的应用程序更容易模块化为子Compose文件。有关包含和使用多个Compose文件的详细信息，请参见使用多个撰写文件。

## 第8步：尝试其他一些命令

- 如果您想在后台运行服务，可以将-d标志（用于“分离”模式）传递给docker compose up，并使用docker compose-ps查看当前正在运行的内容：

```abc
docker compose up -d

Starting composetest_redis_1...
Starting composetest_web_1...

docker compose ps

       Name                      Command               State           Ports     
-------------------------------------------------------------------------------------
composetest_redis_1   docker-entrypoint.sh redis ...   Up      6379/tcp          
composetest_web_1     flask run                        Up      0.0.0.0:8000->5000/tcp
```

- 运行docker compose--help查看其他可用的命令。
- 如果您使用docker Compose up-d启动了Compose，请在完成服务后停止它们：

  ```bash
  $ docker compose stop
  ```
- 使用docker compose down命令，您可以将所有内容都放下，完全移除容器。

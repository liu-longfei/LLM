# Build variables

在Docker Build中，构建参数（ARG）和环境变量（ENV）都是将信息传递到构建过程中的一种手段。您可以使用它们来参数化构建，从而实现更灵活和可配置的构建。

> 警告
> 构建参数和环境变量不适合向构建传递机密，因为它们在最终图像中是公开的。相反，使用秘密装载或SSH装载，它们可以安全地向构建公开秘密。
> 有关详细信息，请参阅构建机密。

## 异同

构建参数和环境变量相似。它们都在Dockerfile中声明，并且可以使用docker build命令的标志进行设置。两者都可以用于对构建进行参数化。但它们各有不同的用途。

### 生成参数

构建参数是Dockerfile本身的变量。使用它们来参数化Dockerfile指令的值。例如，您可以使用生成参数来指定要安装的依赖项的版本。
除非在指令中使用生成参数，否则生成参数对生成没有影响。除非从Dockerfile明确地传递到映像文件系统或配置中，否则它们不可访问或不存在于从映像实例化的容器中。它们可能存在于图像元数据中，如出处证明和图像历史中，这就是为什么它们不适合保密。
它们使Dockerfiles更加灵活，更易于维护。
有关如何使用生成参数的示例，请参见ARG使用示例。

### 环境变量

环境变量被传递到构建执行环境，并保存在从映像实例化的容器中。环境变量主要用于：

- 配置生成的执行环境
- 设置容器的默认环境变量

如果设置了环境变量，则可以直接影响构建的执行以及应用程序的行为或配置。
不能在生成时重写或设置环境变量。环境变量的值必须在Dockerfile中声明。您可以将环境变量和构建参数组合起来，以允许在构建时配置环境变量。
有关如何使用环境变量配置构建的示例，请参见ENV使用示例。

## ARG使用示例

生成参数通常用于指定生成中使用的组件版本，例如映像变体或包版本。
指定版本作为构建参数可以使用不同的版本进行构建，而无需手动更新Dockerfile。它还使维护Dockerfile变得更容易，因为它允许您在文件的顶部声明版本。
构建参数也可以是在多个位置重用值的一种方式。例如，如果你在建筑中使用多种口味的高山，你可以确保你在任何地方都使用相同版本的高山：

- `golang：1.22-alpine${ALPINE_VERSION}`
- `python:3.12-alpine$｛alpine_VERSION｝`
- `nginx:1-alpine$｛alpine_VERSION｝`

以下示例使用构建参数定义node和alpine的版本。

```abc
# syntax=docker/dockerfile:1

ARG NODE_VERSION="20"
ARG ALPINE_VERSION="3.19"

FROM node:${NODE_VERSION}-alpine${ALPINE_VERSION} AS base
WORKDIR /src

FROM base AS build
COPY package*.json ./
RUN npm ci
RUN npm run build

FROM base AS production
COPY package*.json ./
RUN npm ci --omit=dev && npm cache clean --force
COPY --from=build /src/dist/ .
CMD ["node", "app.js"]
```

在这种情况下，构建参数具有默认值。在调用生成时指定它们的值是可选的。要覆盖默认值，您可以使用`--build arg CLI`标志：

```abc
docker build --build-arg NODE_VERSION=current .
```

有关如何使用生成参数的详细信息，请参阅：

- ARG Dockerfile [reference](https://docs.docker.com/reference/dockerfile/#arg)
- docker build --build-arg [reference](https://docs.docker.com/reference/cli/docker/image/build/#build-arg)

### ENV使用示例

使用`ENV`声明环境变量可使该变量可用于构建阶段的所有后续指令。以下示例显示了在使用`npm`安装JavaScript依赖项之前将`NODE_ENV`设置为`production`的示例。设置变量会使`npm`省略仅用于本地开发的包。

```abc
# syntax=docker/dockerfile:1

FROM node:20
WORKDIR /app
COPY package*.json ./
ENV NODE_ENV=production
RUN npm ci && npm cache clean --force
COPY . .
CMD ["node", "app.js"]
```

默认情况下，环境变量在构建时是不可配置的。如果要在构建时更改`ENV`的值，可以组合环境变量和构建参数：

```abc
# syntax=docker/dockerfile:1

FROM node:20
ARG NODE_ENV=production
ENV NODE_ENV=$NODE_ENV
WORKDIR /app
COPY package*.json ./
RUN npm ci && npm cache clean --force
COPY . .
CMD ["node", "app.js"]
```

有了这个Dockerfile，您可以使用`--build-arg`来覆盖`ENV`的默认值：

```abc
$ docker build --build-arg NODE_ENV=development .
```

请注意，由于您设置的环境变量持久存在于容器中，因此使用它们可能会对应用程序的运行时产生意外的副作用。有关如何在生成中使用环境变量的更多信息，请参阅：

- ENV [Dockerfile reference](https://docs.docker.com/reference/dockerfile/#env)

### 范围界定

在Dockerfile的全局范围中声明的构建参数不会自动继承到构建阶段。它们只能在全局范围内访问。

```abc
# syntax=docker/dockerfile:1

# The following build argument is declared in the global scope:
ARG NAME="joe"

FROM alpine
# The following instruction doesn't have access to the $NAME build argument
# because the argument was defined in the global scope, not for this stage.
RUN echo "hello ${NAME}!"
```

本例中的`echo`命令计算结果为`hello ！`因为`NAME`生成参数的值超出了作用域。若要将全局生成参数继承到阶段中，必须使用它们：

```abc
# syntax=docker/dockerfile:1

# Declare the build argument in the global scope
ARG NAME="joe"

FROM alpine
# Consume the build argument in the build stage
ARG NAME
RUN echo $NAME
```

一旦在阶段中声明或使用了构建参数，它就会自动由子阶段继承。

```abc
# syntax=docker/dockerfile:1
FROM alpine AS base
# Declare the build argument in the build stage
ARG NAME="joe"

# Create a new stage based on "base"
FROM base AS build
# The NAME build argument is available here
# since it's declared in a parent stage
RUN echo "hello $NAME!"
```

下图进一步举例说明了构建参数和环境变量继承如何用于多阶段构建。

<img alt="img.png" height="10%" src="img.png" width="10%"/>    

图像链接：https://docs.docker.com/build/images/build-variables.svg
### 预定义的生成参数
本节介绍默认情况下所有生成都可以使用的预定义生成参数。
#### 多平台构建参数
多平台构建参数描述了构建的构建和目标平台。
构建平台是运行构建程序（BuildKit守护进程）的主机系统的操作系统、体系结构和平台变体。
- `BUILDPLATFORM`
- `BUILDOS`
- `BUILDARCH`
- `BUILDVARIANT`

目标平台参数为构建的目标平台保留相同的值，这些值是使用`docker build`命令的`--platform`标志指定的。
- `TARGETPLATFORM`
- `TARGETOS`
- `TARGETARCH`
- `TARGETVARIANT`   

这些参数对于在多平台构建中进行交叉编译非常有用。它们在Dockerfile的全局范围内可用，但不会被构建阶段自动继承。若要在后台文件中使用它们，必须声明它们：
```abc
# syntax=docker/dockerfile:1

# Pre-defined build arguments are available in the global scope
FROM --platform=$BUILDPLATFORM golang
# To inherit them to a stage, declare them with ARG
ARG TARGETOS
RUN GOOS=$TARGETOS go build -o ./exe .
```
For more information about multi-platform build arguments, refer to [Multi-platform arguments](https://docs.docker.com/reference/dockerfile/#automatic-platform-args-in-the-global-scope)
### 代理参数
使用代理生成参数可以指定用于生成的代理。您不需要在Dockerfile中声明或引用这些参数。使用`--build-arg`参数指定代理就足以使您的构建使用该代理。   
默认情况下，代理参数会自动从构建缓存和`docker history`记录的输出中排除。如果您确实引用了Dockerfile中的参数，那么代理配置最终会出现在构建缓存中。   
生成器尊重以下代理生成参数。变量不区分大小写。
- `HTTP_PROXY`
- `HTTPS_PROXY`
- `FTP_PROXY`
- `NO_PROXY`
- `ALL_PROXY`
要为生成配置代理，请执行以下操作：
```abc
docker build --build-arg HTTP_PROXY=https://my-proxy.example.com .
```
For more information about proxy build arguments, refer to [Proxy arguments](https://docs.docker.com/reference/dockerfile/#predefined-args).
### BUILDKIT_COLORS
更改终端输出的颜色。将`BUILDKIT_COORS`设置为以下格式的CSV字符串：
```abc
export BUILDKIT_COLORS="run=123,20,245:error=yellow:cancel=blue:warning=white"
```
颜色值可以是任何有效的RGB十六进制代码，也可以是[BuildKit predefined colors](https://github.com/moby/buildkit/blob/master/util/progress/progressui/colors.go).
将`NO_COLOR`设置为任意值将关闭彩色输出，如所建议的[no-color.org](https://no-color.org/).
使用：
```abc
export BUILDKIT_HOST=tcp://localhost:1234
docker buildx create --name=remote --driver=remote
```
如果同时指定`BUILDKIT_HOST`环境变量和位置参数，则该参数具有优先级。
### BUILDKIT_PROGRESS
设置BuildKit进度输出的类型。有效值为：
- `auto`(default)
- `plain`
- `tty`
- `rawjson`
使用：
```abc
export BUILDKIT_PROGRESS=plain
```
### BUILDKIT_TTY_LOG_LINES
通过将`BUILDKIT_TTY_LOG_LINES`设置为一个数字（默认为6），可以更改tty模式下活动步骤可见的日志行数。
```abc
export BUILDKIT_TTY_LOG_LINES=8
```
### EXPERIMENTAL_BUILDKIT_SOURCE_POLICY
用于指定[BuildKit source policy](https://github.com/moby/buildkit/blob/master/docs/build-repro.md#reproducing-the-pinned-dependencies)文件，用于创建具有固定依赖项的可复制生成。
```abc
export EXPERIMENTAL_BUILDKIT_SOURCE_POLICY=./policy.json
```
示例：
```abc
{
  "rules": [
    {
      "action": "CONVERT",
      "selector": {
        "identifier": "docker-image://docker.io/library/alpine:latest"
      },
      "updates": {
        "identifier": "docker-image://docker.io/library/alpine:latest@sha256:4edbd2beb5f78b1014028f4fbb99f3237d9561100b6881aabbf5acce2c4f9454"
      }
    },
    {
      "action": "CONVERT",
      "selector": {
        "identifier": "https://raw.githubusercontent.com/moby/buildkit/v0.10.1/README.md"
      },
      "updates": {
        "attrs": {"http.checksum": "sha256:6e4b94fc270e708e1068be28bd3551dc6917a4fc5a61293d51bb36e6b75c4b53"}
      }
    },
    {
      "action": "DENY",
      "selector": {
        "identifier": "docker-image://docker.io/library/golang*"
      }
    }
  ]
}
```
### BUILDX_BAKE_GIT_AUTH_HEADER
> Introduced in Buildx version 0.14.0

在私有Git存储库中使用远程Bake定义时设置HTTP身份验证方案。这相当于`GIT_AUTH_HEADER` [scret](https://docs.docker.com/build/building/secrets/#http-authentication-scheme)，但在加载远程烘焙文件时有助于烘焙中的飞行前身份验证。支持的值有`bearer`（默认值）和`basic`。
使用：
```abc
export BUILDX_BAKE_GIT_AUTH_HEADER=basic
```
### BUILDX_BAKE_GIT_AUTH_TOKEN
> Introduced in Buildx version 0.14.0

在私有Git存储库中使用远程Bake定义时设置HTTP身份验证令牌。这相当于`GIT_AUTH_TOKEN`[scret](https://docs.docker.com/build/building/secrets/#git-authentication-for-remote-contexts)，但在加载远程Bake文件时有助于Bake中的飞行前身份验证。
使用：
```abc
export BUILDX_BAKE_GIT_AUTH_TOKEN=$(cat git-token.txt)
```
### BUILDX_BAKE_GIT_SSH
> Introduced in Buildx version 0.14.0

用于指定SSH代理套接字文件路径的列表，以便在专用存储库中使用远程Bake定义时转发给Bake以向Git服务器进行身份验证。这类似于用于构建的SSH装载，但在解析构建定义时有助于Bake中的飞行前身份验证。   
通常不需要设置此环境，因为默认情况下Bake将使用`SSH_AUTH_SOCK`代理套接字。如果要使用具有不同文件路径的套接字，则只需要指定此变量。此变量可以使用逗号分隔的字符串获取多个路径。
使用：
```abc
export BUILDX_BAKE_GIT_SSH=/run/foo/listener.sock,~/.creds/ssh.sock
```
### BUILDX_BUILDER
覆盖已配置的生成器实例。与`docker buildx --builder` CLI标志。
使用：
```abc
export BUILDX_BUILDER=my-builder
```
### BUILDX_CONFIG
可以使用`BUILDX_CONFIG`指定用于生成配置、状态和日志的目录。此目录的查找顺序如下：
- `$BUILDX_CONFIG`
- `$DOCKER_CONFIG/buildx`
- `~/.docker/buildx (default)`
使用：
```abc
export BUILDX_CONFIG=/usr/local/etc
```
### BUILDX_EXPERIMENTAL
Enables experimental build features.
使用：
```abc
export BUILDX_EXPERIMENTAL=1
```
### BUILDX_GIT_CHECK_DIRTY
> Introduced in Buildx version 0.10.4

When set to true, checks for dirty state in source control information for [provenance attestations](https://docs.docker.com/build/attestations/slsa-provenance/).
使用：
```abc
export BUILDX_GIT_CHECK_DIRTY=1
```
### BUILDX_GIT_INFO
> Introduced in Buildx version 0.10.0

When set to false, removes source control information from [provenance attestations](https://docs.docker.com/build/attestations/slsa-provenance/).
使用：
```abc
export BUILDX_GIT_INFO=0
```
### BUILDX_GIT_LABELS
> Introduced in Buildx version 0.10.0

Adds provenance labels, based on Git information, to images that you build. The labels are:
- `com.docker.image.source.entrypoint`: Location of the Dockerfile relative to the project root
- `org.opencontainers.image.revision`: Git commit revision
- `org.opencontainers.image.source`: SSH or HTTPS address of the repository

示例：
```abc
  "Labels": {
    "com.docker.image.source.entrypoint": "Dockerfile",
    "org.opencontainers.image.revision": "5734329c6af43c2ae295010778cd308866b95d9b",
    "org.opencontainers.image.source": "git@github.com:foo/bar.git"
  }
```
使用：
- Set ·BUILDX_GIT_LABELS=1· to include the `entrypoint` and `revision` labels.
- Set `BUILDX_GIT_LABELS=full` to include all labels.

If the repository is in a dirty state, the `revisio`n gets a `-dirty` suffix.
### BUILDX_NO_DEFAULT_ATTESTATIONS
> Introduced in Buildx version 0.10.4

By default, BuildKit v0.11 and later adds [provenance attestations](https://docs.docker.com/build/attestations/slsa-provenance/) to images you build. Set `BUILDX_NO_DEFAULT_ATTESTATIONS=1` to disable the default provenance attestations.
使用：
```abc
export BUILDX_NO_DEFAULT_ATTESTATIONS=1
```
### BUILDX_NO_DEFAULT_LOAD
When you build an image using the `docker` driver, the image is automatically loaded to the image store when the build finishes. Set `BUILDX_NO_DEFAULT_LOAD` to disable automatic loading of images to the local container store.
使用：
```abc
export BUILDX_NO_DEFAULT_LOAD=1
```

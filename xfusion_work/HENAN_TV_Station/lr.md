### 镜像源地址

-i https://pypi.tuna.tsinghua.edu.cn/simple
-i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

-i http://70.182.56.39:8081/repository/pypi-xfusion/simple --trusted-host 70.182.56.39:8081 

http\_proxy http://proxy.xfusion.com:8080
https\_proxy http://proxy.xfusion.com:8080

no\_proxy 70.182.56.16,70.182.56.3,localhost,70.182.0.0/16,127.0.0.1,\*.local,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16

REQUESTS_CA_BUNDLE = D:\data\software\Xfusion_Rootcert.crt

export REQUESTS_CA_BUNDLE=D:\data\software\Xfusion_Rootcert.crt

export REQUESTS_CA_BUNDLE=/mnt/d/data/software/Xfusion_Rootcert.crt

export http\_proxy=http://proxy.xfusion.com:8080
export https\_proxy=http://proxy.xfusion.com:8080

setx http\_proxy=http://proxy.xfusion.com:8080
setx https\_proxy=http://proxy.xfusion.com:8080



export http_proxy=192.168.102.148:8080 
export https_proxy=192.168.102.148:8080 
export SSL_CERT_FILE=/root/Xfusion.crt 
export REQUESTS_CA_BUNDLE=/root/Xfusion.crt



export REQUESTS_CA_BUNDLE=/home/W0016778/software/Xfusion_Rootcert.crt

CUDA_VISIBLE_DEVICES=5 

### 黄区虚拟机conda create 创建新环境的方法：

```context
* 配置小网代理
* 配置证书 。 cat Xfusion_Rootcert.crt >> /etc/pki/tls/certs/ca-bundle.crt
* 配置环境变量。export CONDA_SSL_VERIFY=false
* 不要用清华源，直接conda create
```

### 杀死python进程

killall -9 python

### ssl错误

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

### 加入证书

```python
import os
os.environ['REQUESTS_CA_BUNDLE'] = r"D:\data\software\Xfusion_Rootcert.crt"
```

本地电脑添加证书

```
https://help.aliyun.com/zh/ssl-certificate/user-guide/how-to-install-root-and-intermediate-certificates-on-windows-systems
```

### 英语

| 单词            | 音标                                  | 意思                                                                                                                                                                                    |
| ------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| manual        | 英 /ˈmænjuəl/ 美 /ˈmænjuəl/           | n. 手册;说明书;指南<br /> adj. 手动的;手工的;体力的;用手的;手控的;用手操作的                                                                                                                                     |
| replacement   | 英 /rɪˈpleɪsmənt/ 美 /rɪˈpleɪsmənt/   | n. 更换;替换;替代品;替代者;(尤指工作中的)接替者;替换物                                                                                                                                                      |
| scope         | 英 /skəʊp/ 美 /skoʊp/                 | n.(题目、组织、活动等的)范围;(做或实现某事的)机会，能力;…镜(观察仪器)<br />vt. 仔细看;彻底检查                                                                                                                            |
| latent        | 英 /ˈleɪtnt/ 美 /ˈleɪtnt/             | adj. 潜伏的，潜在的，隐藏的，不易察觉的;<心>潜在的，隐性的;<医>潜伏的，隐性的;<植>休眠的，潜伏的<br />n.(犯罪现场等的)隐约指纹，潜指印                                                                                                       |
| slider        | 英 /ˈslaɪdə(r)/ 美 /ˈslaɪdər/         | n.(控制收音机音量等的)滑杆，滑动器;浮动块，滚动条滑块;红腹彩龟(北美淡水龟)                                                                                                                                             |
| interface     | 英 /ˈɪntəfeɪs/ 美 /ˈɪntərfeɪs/        | n. 接口;(人机)界面(尤指屏幕布局和菜单);接口程序;连接电路;(两学科、体系等的)接合点，边缘区域<br />v.(使通过界面或接口)接合，连接                                                                                                           |
| agent         | 英 /ˈeɪdʒənt/ 美 /ˈeɪdʒənt/           | n.(演员、音乐家、运动员、作家等的)代理人;（化学）剂;(企业、政治等的)经纪人;动因(指对事态起重要作用的人、事物);原动力;施事者                                                                                                                  |
| issues        | 英 /ˈɪʃuːz/ 美 /ˈɪʃuːz/               | n.(有关某事的)问题，担忧;一期;重要议题;争论的问题;期号<br />v. 发出;(正式)发给，供给;公布;宣布;(尤指通过正式文件)将…诉诸法律 issue的第三人称单数和复数                                                                                           |
| entity        | 英 /ˈentəti/ 美 /ˈentəti/             | n. 实体;独立存在物                                                                                                                                                                           |
| massive       | 英 /ˈmæsɪv/ 美 /ˈmæsɪv/               | adj. 大量的，巨大的，大规模的;〈地〉块状的，（成岩层）以厚层出现的;<br />〈医〉大块的，大范围的，大剂量的;大而重的，大块的，厚实的，结实的，魁梧的;<br />（面貌、五官等）粗大的，宽广的;给人深刻印象的，雄伟的，庄严的;（疾病）非常严重的，十分危急的;〈矿〉非晶质的，均匀构造的 <br />n. 〈英口〉（来自特定地区的）嘻哈（或丛林）音乐迷 |
| dialect       | 英 /ˈdaɪəlekt/ 美 /ˈdaɪəlekt/         | n. 方言;土话;地方话                                                                                                                                                                          |
| schema        | 英 /ˈskiːmə/ 美 /ˈskiːmə/             | n.(计划或理论的)提要，纲要                                                                                                                                                                       |
| casual        | 英 /ˈkæʒuəl/ 美 /ˈkæʒuəl/             | adj. 随便的;临时的;偶然的;非正式;不经意的;漫不经心的;马虎的;不在乎的;无忧无虑的;疏忽的;感情不深的<br />n. 临时工;便装;便鞋                                                                                                            |
| awesome       | 英 /ˈɔːsəm/ 美 /ˈɔːsəm/               | adj. 很好的(或极好玩的等);令人惊叹的;很困难的;使人惊惧的;难得吓人的                                                                                                                                               |
| discrete      | 英 /dɪˈskriːt/ 美 /dɪˈskriːt/         | adj. 离散的;分离的;各别的;互不相连的                                                                                                                                                                |
| ensemble      | 英 /ɒnˈsɒmbl/ 美 /ɑːnˈsɑːmbl/         | n. 合奏，合唱;（经常在一起演出的小型）乐团，剧团，舞剧团;全体，整体;成套的东西 adv. 一起，同时                                                                                                                                 |
| indicator     | 英 /ˈɪndɪkeɪtə(r)/ 美 /ˈɪndɪkeɪtər/   | n. 指示器;标志;迹象;指针;指示信号;转向灯;方向灯                                                                                                                                                          |
| exponential   | 英 /ˌekspəˈnenʃl/ 美 /ˌekspəˈnenʃl/   | adj. 指数的;幂的;越来越快的;由指数表示的 n. 指数                                                                                                                                                        |
| accumulate    | 英 /əˈkjuːmjəleɪt/ 美 /əˈkjuːmjəleɪt/ | v. 积累;积聚;(数量)逐渐增加;(数额)逐渐增长                                                                                                                                                            |
| Proximal      | 英 /ˈprɒksɪməl/ 美 /ˈprɑːksɪməl/      | 近端;近端的;近侧;最接近;近心                                                                                                                                                                      |
| copilot       | 英 /ˈkəʊˌpaɪlət/ 美 /ˈkoʊˌpaɪlət/     | n. （飞机的）副驾驶员;缩写词为coplt                                                                                                                                                                |
| collaborative | 英 /kəˈlæbərətɪv/ 美 /kəˈlæbəreɪtɪv/  | adj. 合作的;协作的;协力的                                                                                                                                                                      |
| Ablation      | 英/əˈbleɪʃn/ 美/ə'bleɪʃn/             | n.消融，切除；冲蚀；消蚀；磨削；                                                                                                                                                                     |
| exact         | 英[ɪgˈzækt] 美[ɪɡˈzækt]               | adj.准确的；严密的；精密的，精确的； vt.要求；苛求；迫使；强求；                                                                                                                                                  |
| inductive     | 英 /ɪnˈdʌktɪv/ 美 /ɪnˈdʌktɪv/         | adj. 归纳的;归纳法的;电感应的                                                                                                                                                                    |
| casual        | 英 /ˈkæʒuəl/ 美 /ˈkæʒuəl/             | adj. 随便的;临时的;偶然的;非正式;不经意的;漫不经心的;马虎的;不在乎的;无忧无虑的;疏忽的;感情不深的<br />n. 临时工;便装;便鞋                                                                                                            |

# Nvicate删除注册表试用

https://www.cnblogs.com/smj-7038/p/17560479.html

# 贝叶斯公式

![img.png](img.png)

## python限定用哪些显卡

1、命令行运行代码
CUDA_VISIBLE_DEVICES=0,1,2,3

2、 export CUDA_VISIBLE_DEVICES=5,6

## 命令行中打断点

```python
import pdb
pdb.set_trace()
```

# transformers

## HfArgumentParser

[transformer.HfArgumentParser案例以及Debug-CSDN博客](https://blog.csdn.net/sjxgghg/article/details/131229480)

## dataclass和field

[一文搞懂dataclass和field-CSDN博客](https://blog.csdn.net/raelum/article/details/136680946)

## AutoTokenizer

https://juejin.cn/post/7350510977052229668

## nltk

C:\Users\LW0016778\AppData\Roaming\nltk_data
/home/software/nltk_data /root/nltk_data/

## cl100

/home/software/data-gym-cache /tmp/data-gym-cache

# torch

## torch.cumsum()

> 函数作用：在张量的某个维度上进行累加,从语言描述上比较抽象，看下以下代码加以说明（针对二维张量）：

x = np.array([[1, 2, 3], [4, 5, 6]])
x = torch.from_numpy(x)
x_ = x.cumsum(dim=0)
tensor([[1, 2, 3],
[5, 7, 9]])

## torch.multinomial()

#### 以输入的张量作为权重进行多分布采样

import torch
input = torch.Tensor([5, 0, 10, 0])
#无放回
output = torch.multinomial(input, num_samples=4)
print(output)
#有放回
output = torch.multinomial(input, num_samples=4, replacement=True)
print(output)

#### 输出

tensor([2, 0, 3, 1])
tensor([2, 2, 2, 2])

## torch.triu()

triu = triangle up （我猜的）
顾名思义，这个函数的作用相同，都是返回上三角矩阵，定义分别如下：
这两个函数的对应参数都是相同的含义。m 代表要操作的矩阵，k 代表以哪条对角线作为上三角矩阵的对角线。k=0 代表主对角线，k 为正数则从主对角线开始向上数第 k 条，k 为负数则从主对角线开始向下数第 k 条。

# 大模型微调

## 大模型学习资料

https://github.com/THUDM/ChatGLM-6B/tree/main/ptuning
https://github.com/shibing624/MedicalGPT
https://huggingface.co/docs/transformers/index
https://github.com/huggingface/peft
https://transformers.run/

## 大模型中的Agent

在大型模型的背景下，Agent可以理解为一种能够自主理解、规划和执行复杂任务的系统。
那么，什么样的应用才能算作一个Agent应用呢？OpenAI的华人科学家翁丽莲给出了一个直观的定义：“Agent = 大型模型 + 记忆 + 主动规划 + 工具使用”。

## ChatGLM-6b

代码地址：https://github.com/THUDM/ChatGLM-6B/tree/main/ptuning
模型地址：https://huggingface.co/THUDM/chatglm-6b/tree/main

## glm-6b-baes

https://huggingface.co/THUDM/chatglm3-6b-base/tree/main

## 序列标注任务

常见的序列标注任务有命名实体识别 NER (Named Entity Recognition) 和词性标注 POS (Part-Of-Speech tagging)。
命名实体识别 NER 旨在识别出文本中诸如人物、地点、组织等实体，即为所有的 token 都打上实体标签（包含“非实体”）。词性标注 POS 旨在为文本中的每一个词语标注上对应的词性，例如名词、动词、形容词等。

# 图像生成

## midjourney

### 服务器文件夹复制到本机文件夹

scp -r root@70.182.40.130:/data/transforms/chatglm-6b ./

## 邮箱密码

rB4X#verzKn%zw

# vscode

## vscode安装Fira Code字体

"editor.fontFamily": "Fira Code",
"editor.fontLigatures": true,

## vscode离线下载插件

https://marketplace.visualstudio.com/VSCode
字体：'Cascadia Code', 'Fira Code'
插件：Material Icon Theme、Remote、Python、Markdown Preview Enhanced

## VsCode配置ssh免密远程连接服务器

https://blog.csdn.net/weixin_46773333/article/details/128335416

## ssh设置免密登录后登录仍需密码

https://developer.aliyun.com/article/1169754

## windows本机配置代理

70.182.*.*

## streamlit run *.py

demo.queue().launch(share=False, inbrowser=True, server_name='0.0.0.0')

## CondaSSLError

conda config --set ssl\_verify false

# tmux

## tmux删除会话中的一个窗口

Ctrl + b + x

## lsof命令查看端口

sudo lsof -i:{port}
lsof代表list open files，是一个用于查看系统打开的文件的命令行工具。在Ubuntu中，可以通过命令lsof -i:{port}查看对应端口的进程信息。
其中，{port}需要替换成具体的端口号。例如，若要查看端口号为8080的进程信息，可以使用命令sudo lsof -i:8080。需要注意的是，使用lsof命令需要root权限。

## 服务器使用streamlit命令，跳转

tokenizer = AutoTokenizer.from_pretrained("/data/transformers/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("/data/transformers/chatglm3-6b", trust_remote_code=True).cuda()

demo.queue().launch(share=False, inbrowser=True, server_name='0.0.0.0')

## 大模型评测数据集(MMLU、MMCU、C-Eval)

## 量化等级(4bit，8-bit)

## 怎么加载魔板

flashAttention-2

## SQL在线练习

https://www.w3schools.com/sql/trysql.asp?filename=trysql_op_in

## DU

### 显示当前文件夹下的所有文件的大小：

```commandline
du -h
```

### 「du」命令则可以统计当前文件夹内的所有文件的大小，命令如下：

```
du -sh *
```

### 查看指定文件夹大小：

```commandline
du -sh path/
```

## Anaconda打包镜像

https://blog.csdn.net/qq_45634934/article/details/131741139

## pycharm破解

链接：https://hardbin.com/ipfs/bafybeia4nrbuvpfd6k7lkorzgjw3t6totaoko7gmvq5pyuhl2eloxnfiri/
防火墙禁止软件联网：https://www.win10h.com/jiaocheng/17657.html

## sql语句学习

https://blog.csdn.net/m0_71741835/article/details/127624612

## pandas单元格值是否为NaN的判断方法

https://blog.csdn.net/jiboli123/article/details/131162857

## json加载SQL查询数据库的结果

https://www.cnblogs.com/kylexy/p/17779599.html

## DataFrame交换两列

https://blog.csdn.net/qq_52079156/article/details/120536921

## 官方提示工程指南

https://baoyu.io/translations/openai/openai-prompt-engineering-guides
https://www.promptingguide.ai/zh/techniques/knowledge

## RAG

https://baoyu.io/translations/rag/advanced-rag-techniques-an-illustrated-overview?continueFlag=e6f72bb74036e2105604dd6c9a032483

## GIT代码提交最佳实践

http://w3.xfusion.com/sns/topic/sns_topic_main/snsTopicMain.do?method=view&fdTabId=18bf602028379f562b2510a4ab5b18d2&fdId=18cab46b7fae2762425a6c14bfaa9a80&s_css=default

[GIt保持远程 源仓库与Fork仓库同步--2017年6月13日 - jingbostar - 博客园 (cnblogs.com)](https://www.cnblogs.com/jingbostar/p/7001913.html)

# GIT

## git rebase

教程：https://blog.csdn.net/weixin_42310154/article/details/119004977

[git rebase （合并代码和整理提交记录）图文详解-CSDN博客](https://blog.csdn.net/wangdawei_/article/details/131669124)

[git reset 命令详解 git revert命令详解。-CSDN博客](https://blog.csdn.net/wangdawei_/article/details/124543824)

### 流程

git remote add remote ssh://git@git.codehub.xfusion.com:2222/XI_Force/LLM_Solution/General_Application/Document_Review_Application.git

git pull remote

git chekcout feature
git rebase master
解决冲突
git add <冲突文件名>
git rebase --continue即可
git merge 或者 git push(具体还不清楚，需要后面确认)

## git stash

`git stash` 是一个非常有用的 Git 命令，它允许你临时保存你的工作进度，而不必提交不完整的代码到版本库中。这在你需要快速切换分支处理其他事务，但当前分支的工作尚未完成到可以提交的程度时特别有用。

### 基本使用方法

1. **保存更改**：
   使用 `git stash` 或 `git stash push` 将当前工作目录和暂存区的更改保存起来。你可以为 stash 提供一个描述，以便于以后识别：
   `git stash push -m "描述信息"`

2. **查看已存储的 stash 列表**：
   `git stash list`
   这会显示所有存储的 stash，每个 stash 都有一个索引，从最新的开始编号，如 `stash@{0}`, `stash@{1}` 等。

3. **应用最近的 stash**：
   `git stash apply`
   这会应用最近的 stash 更改到当前工作目录，但不会从 stash 列表中删除这些更改。如果你想应用特定的 stash，可以指定它的索引：
   `git stash apply stash@{1}`

4. **弹出最近的 stash**：
   `git stash pop`
   这与 `apply` 类似，但它会在应用 stash 后从存储的 stash 列表中删除这个 stash。

5. **删除一个特定的 stash**：
   `git stash drop stash@{0}`
   这会删除指定的 stash。

6. **清空所有的 stash**：
   `git stash clear`

### 高级用法

* **包括未跟踪的文件**：
  默认情况下，`git stash` 不会包括未跟踪的文件（新文件）。如果你想包括这些文件，可以使用：
  `git stash push -u`
  或者
  `git stash push --include-untracked`

* **包括忽略的文件**：
  如果你还想包括被 `.gitignore` 忽略的文件，可以使用：
  `git stash push -a`
  或者
  `git stash push --all`

### 使用场景示例

假设你正在开发一个功能，但突然需要修复一个紧急的 bug。你可以：

1. 使用 `git stash` 保存当前进度。
2. 切换到需要修复 bug 的分支。
3. 修复 bug 并提交更改。
4. 切换回原来的分支。
5. 使用 `git stash pop` 恢复你的工作进度并继续之前的工作。

通过这种方式，`git stash` 使得在不同任务之间切换变得无缝且安全。

## prompt工程

https://www.promptingguide.ai/zh/techniques/knowledge

## 指示学习（Instruct Learning）和提示（Prompt Learning）学习指示学习（Instruct Learning）和提示（Prompt Learning）学习

指示学习和提示学习的目的都是去挖掘语言模型本身具备的知识。不同的是Prompt是激发语言模型的**补全能力**，例如根据上半句生成下半句，或是完形填空等。Instruct是激发语言模型的理解能力，它通过给出更明显的指令，让模型去做出正确的行动。

pairwise ranking loss

## 手把手教你在昇腾平台上搭建PyTorch训练环境：

https://www.bilibili.com/read/cv23917714/

## 昇腾教程

[在线课程-昇腾社区 (hiascend.com)](https://www.hiascend.com/edu/courses?activeTab=PyTorch%E5%A4%A7%E6%A8%A1%E5%9E%8B%E8%BF%81%E7%A7%BB%E8%B0%83%E4%BC%98)

## 昇腾社区

[昇腾社区-官网丨昇腾万里 让智能无所不及 (hiascend.com)](https://www.hiascend.com/)

## gitee昇腾

[Ascend: 昇腾万里，让智能无所不及 (gitee.com)](https://gitee.com/ascend)

## 昇腾模型库

[ModelZoo-昇腾社区 (hiascend.com)](https://www.hiascend.com/zh/software/modelzoo/models?tag=pytorch)

## 查看Linux服务器架构是x86还是arm

uname -m

## 昇腾环境安装

https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes/ptes_00028.html

## 昇腾环境安装

[资源下载中心-昇腾社区 (hiascend.com)](https://www.hiascend.com/zh/developer/download/community/result?module=cann)

## 查看系统版本

cat /etc/os-release

## fastGPT文档教程

[接入谷歌搜索 | FastGPT](https://doc.fastgpt.in/docs/workflow/examples/google_search/#%E5%86%99%E4%B8%80%E4%B8%AA-google-search-%E6%8E%A5%E5%8F%A3)

昇腾推理

[昇腾平台llm微调与推理 (xfusion.com)](http://w3.xfusion.com/sns/topic/sns_topic_main/snsTopicMain.do?method=view&fdId=18dd5be2beec2dc53d7994040eca9741)

[昇腾环境安装 (xfusion.com)](https://w3.xfusion.com/sns/topic/sns_topic_main/snsTopicMain.do?method=view&fdTabId=18bf602028379f562b2510a4ab5b18d2&fdId=18d1bf4d121e1088cfaa943410d84813&s_css=default)

## 两个高斯分布之间的KL散度

https://blog.csdn.net/qq_33552519/article/details/130561606

## 期望和方差

https://www.bilibili.com/video/BV16f4y1D7Yf/?vd_source=fffd0a55d832fc312ff82010c51836f8

## linux给普通用户增加权限

https://blog.csdn.net/sinat_36118270/article/details/62899093

### 删除所有容器的命令

docker rm -f $(docker ps -a -q)

## wsl导入迁移镜像

https://blog.csdn.net/weixin_63133658/article/details/134533272

## diffusion讲解

https://aistudio.baidu.com/projectdetail/4867936

## 查看端口号占用

lsof -i:端口号

## 根据进程查看具体信息

ps -f -p PID

-p指定进程，-f 输出所有信息

kill PID # 杀死对应的进程

# Doceker

## docker镜像打包和加载

> docker save -o images_tar_image.tar images:tag  # 打包镜像

> docker load -i images_tar_image.tar  # 加载镜像

## docker删除特定镜像的容器

> 查看看特定镜像穿件的容器：sudo docker ps -a --filter ancestor=bank:v1.0

> 删除所有容器: docker rm -f $(docker ps -a -q)

### docker运行挂载硬盘

sudo docker run -v /home/W0016778/code/demo_app/tmp_datasets:/usr/cr/app/tmp_datesets -it bank:v1.6 /bin/bash

## sed对新环境编辑文件

例如：要使用 `sed` 将文件中的 `127.0.0.1` 替换为 `0.0.0.0`，您可以使用以下命令：

> sed -i 's/127.0.0.1/0.0.0.0/g' filename

- `-i` 选项表示直接在原始文件上进行修改。
* `'s/127.0.0.1/0.0.0.0/g'` 是 `sed` 命令的替换操作。其中，`s` 表示替换操作的开始，`127.0.0.1` 是要被替换的内容，`0.0.0.0` 是替换后的内容，`g` 表示全局替换，即一行中所有匹配的内容都会被替换。

* `filename` 是要进行替换操作的文件名。

### docker run加入代理

docker run -e "http_proxy=http://192.168.185.35:8080" -e "https_proxy=http://192.168.185.35:8080" -it bank:v1.5 /bin/bash

### docker build加入代理

docker build .
--build-arg "HTTP_PROXY=http://192.168.185.35:8080/"
--build-arg "HTTPS_PROXY=http://192.168.185.35:8080/"
--build-arg "NO_PROXY=localhost,127.0.0.1,.example.com"
-t bank:v1.8

## 安装docker-compose

https://blog.51cto.com/u_13171517/7608668

## docker中文参考手册

[Docker — 从入门到实践 | Docker 从入门到实践 (docker-practice.com)](https://vuepress.mirror.docker-practice.com/)

[GitHub - yeasy/docker\_practice: Learn and understand Docker&Container technologies, with real DevOps practice!](https://github.com/yeasy/docker_practice)

[前言 - Docker — 从入门到实践 (gitbook.io)](https://yeasy.gitbook.io/docker_practice/)

## pdfplumber

github 官方仓库:https://github.com/jsvine/pdfplumber

PDFPlumber使用入门：https://blog.csdn.net/weixin_48629601/article/details/107224376

gitcode 官方仓库：https://gitcode.com/jsvine/pdfplumber/overview?utm_source=artical_gitcode#extracting-tables



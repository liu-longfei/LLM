# LLM

## 生成扩散模型

https://spaces.ac.cn/archives/9119

## 浏览器翻译插件（沉浸式翻译）
https://immersivetranslate.com/    
https://immersivetranslate.com/zh-Hans/

## 书籍
https://github.com/LLMBook-zh/LLMBook-zh.github.io

## 免费GPT网站

https://github.com/LiLittleCat/awesome-free-chatgpt/blob/main/README_en.md   
https://ai.ywzxkj.com/   
https://chat.tinycms.xyz:3002/#/chat   
https://www.freegpt.pl/?from=tz   
https://www.freegpt.pl/?from=tongzhi   
https://freegpt.es/?from=tongzhi   
https://gpt4o.so/zh-CN （开无痕模式可以一直用GPT 4o，写作必备）   

## 好用字体

'Cascadia Code', 'Fira Code'

## poetry

https://notes.zhengxinonly.com/environment/use-poetry.html

# Ubuntu
## 安装中文输入法
```bash
1、sudo apt install ibus-rime
2、设置->键盘->输入源->添加输入源->选择中文（rime）->添加
3、改为横向展示： vim ~/.config/ibus/rime/build/ibus_rime.yaml
4、horizontal: true
5、重启：ibus restart
```

# 数据集

## 1.预训练数据集

悟道数据集

## 2.指令微调数据集：

https://huggingface.co/datasets/BAAI/COIG
https://huggingface.co/datasets/BelleGroup/school_math_0.25M
https://huggingface.co/datasets/shibing624/alpaca-zh
https://huggingface.co/datasets/llamafactory/alpaca_zh
https://huggingface.co/datasets/llm-wizard/alpaca-gpt4-data-zh

## 3.训练奖励模型数据集

https://huggingface.co/datasets/dikw/hh_rlhf_cn
https://huggingface.co/datasets/liyucheng/zhihu_rlhf_3k

## 4.评估数据集

# 可能得数据集网址

https://www.cluebenchmarks.com/dataSet_search.html
https://github.com/sunzeyeah/RLHF

## 医疗领域大模型数据

https://huggingface.co/datasets/michaelwzhu/ChatMed_Consult_Dataset
https://huggingface.co/datasets/michaelwzhu/ShenNong_TCM_Dataset
https://github.com/FutureForMe/Chinese_Medical_Natural_Language_Processing_Resources_and_Papers
https://github.com/shibing624/MedicalGPT

## api_key
### SerpAPIWrapper
from langchain import SerpAPIWrapper
为了使用谷歌搜索（SERPAPI）， WolframAlpha，您需要自行申请它们的 API KEY，然后填入此处
os.environ['SERPAPI_API_KEY'] = '9a860ad90aa50f9428933bb26c636e7cceb7155133e8a51dcb2d6d460df4d9ba'
os.environ['WOLFRAM_ALPHA_APPID'] = 'RH4ERE-VHKHLVUA8A'


### tavily:
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

### langchain:
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

### Google Serper
https://serper.dev/dashboard
os.environ["SERPER_API_KEY"] = "ce8271080378a939bd4eadc024b81c1a339bcca1"

### Google Cloud
GOOGLE_API_KEY: AIzaSyDHIO8co1cAIJJRsswIm3oKq8q--Bkglv8
GOOGLE_CSE_ID: https://cse.google.com/cse.js?cx=c0df220e7715047a1


## Ollama部署大模型
https://www.53ai.com/news/qianyanjishu/1196.html

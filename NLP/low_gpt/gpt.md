# 生成式任务的预训练和推理流程

## 输入数据

LLM的输入数据是一段文本，可以是一个句子或一段话。文本通常被表示成单词或字符的序列。

`[君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。...五花马、千金裘，呼儿将出换美酒，与尔同销万古愁]`

## Tokenization

之后需要将文本进行Tokenization，将其切分成单词或字符，形成Token序列。之后再将文本映射成模型可理解的输入形式，将文本序列转换为整数索引序列(这个索引就是单词或字符在语料库中的index)，这个过程通常由一些开源的文本Tokenzier工具，如sentencepiece等来处理

序列化:

`['BOS','君','不','见','黄','河','之','水','天','上','来','，' ,'奔','流','到'...'与','尔','同','销','万','古','愁','EOS']`

假设语料库索引化

`->['BOS','10','3','67','89','21','45','55','61','4','324','565' ,'789','6567','786'...'7869','9','3452','563','56','66','77','EOS']`

## Embedding
文本信息经过Tokenization之后变成了token序列，而Embedding则继续将每个Token映射为一个实数向量，为Embeding Vector

> 'BOS'-> [p<sub>00</sub>,p<sub>01</sub>,p<sub>02</sub>,...,p<sub>0d-1</sub>]
> '10' -> [p<sub>10</sub>,p<sub>11</sub>,p<sub>12</sub>,...,p<sub>1d-1</sub>]
> '3'  -> [p<sub>20</sub>,p<sub>21</sub>,p<sub>22</sub>,...,p<sub>2d-1</sub>]
> ...
> 'EOS'-> [p<sub>n0</sub>,p<sub>n1</sub>,p<sub>n2</sub>,...,p<sub>nd-1</sub>]

## 位置编码
对于Token序列中的每个位置，添加位置编码（Positional Encoding）向量，以提供关于Token在序列中位置的信息。位置编码是为了区分不同位置的Token，并为模型提供上下文关系的信息。

> 'BOS'-> [p<sub>00</sub>,p<sub>01</sub>,p<sub>02</sub>,...,p<sub>0d-1</sub>]      [pe<sub>00</sub>,pe<sub>01</sub>,pe<sub>02</sub>,...,pe<sub>0d-1</sub>]
> '10' -> [p<sub>10</sub>,p<sub>11</sub>,p<sub>12</sub>,...,p<sub>1d-1</sub>]      [pe<sub>10</sub>,pe<sub>11</sub>,pe<sub>12</sub>,...,pe<sub>1d-1</sub>]
> '3'  -> [p<sub>20</sub>,p<sub>21</sub>,p<sub>22</sub>,...,p<sub>2d-1</sub>]  +   [pe<sub>20</sub>,pe<sub>21</sub>,pe<sub>22</sub>,...,pe<sub>2d-1</sub>]
> ...
> 'EOS'-> [p<sub>n0</sub>,p<sub>n1</sub>,p<sub>n2</sub>,...,p<sub>nd-1</sub>]      [pe<sub>n0</sub>,pe<sub>n1</sub>,pe<sub>n2</sub>,...,pe<sub>nd-1</sub>]

## Transformer
在生成任务中，模型只需要用到Transformer 的decoder阶段，即Decoder-Only，比如GPT、LLaMA 都是。

## 输出处理
生成的Token序列通过一个输出层，通常是线性变换加上Softmax函数，将每个位置的概率分布转换为对应Token的概率。根据概率，选择概率最高的Token或者作为模型的预测结果。或者其他的的方法生成next token

## 自回归生成
在生成任务中，使用自回归（Autoregressive）方式，即逐个生成输出序列中的每个Token。在解码过程中，每次生成一个Token时，使用前面已生成的内容作为上下文，来帮助预测下一个Token。

```

In  [1]: {prompt:"将进酒："}
Out [1]: 将进酒：人

In  [2]: 将进酒：人
Out [2]: 将进酒：人生

In  [3]: 将进酒：人生
Out [3]: 将进酒：人生得

In  [4]: 将进酒：人生得
Out [4]: 将进酒：人生得意

In  [5]: 将进酒：人生得意
Out [5]: 将进酒：人生得意需


In  [6]: 将进酒：人生得意需
Out [6]: 将进酒：人生得意需尽

In  [7]: 将进酒：人生得意需尽
Out [7]: 将进酒：人生得意需尽欢
```

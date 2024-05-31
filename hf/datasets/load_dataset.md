# Process

数据集提供了许多用于修改数据集的结构和内容的工具。这些工具对于整理数据集、创建额外的列、在功能和格式之间转换等等都很重要。

本指南将向您展示如何：

- 重新排列行并拆分数据集。
- 重命名和删除列，以及其他常见的列操作。将处理函数应用于数据集中的每个示例。
- 连接数据集。
- 应用自定义格式转换。
- 保存和导出已处理的数据集。

有关处理其他数据集模式的更多详细信息，请参阅流程音频数据集指南、流程图像数据集指南或流程文本数据集指南。

本指南中的示例使用MRPC数据集，但请随意加载您选择的任何数据集并遵循！

```python
from datasets import load_dataset
dataset = load_dataset("glue", "mrpc", split="train")
```

> 本指南中的所有处理方法都返回一个新的数据集对象。修改未到位。要小心覆盖以前的数据集！

## Sort, shuffle, select, split, and shard

有几个函数用于重新排列数据集的结构。这些函数对于只选择所需的行、创建训练和测试拆分以及将非常大的数据集分割成更小的块非常有用。

### Sort

使用sort()根据列值的数值对列值进行排序。提供的列必须与NumPy兼容。

```python
>>> dataset["label"][:10]
[1, 0, 1, 0, 1, 1, 0, 1, 0, 0]
>>> sorted_dataset = dataset.sort("label")
>>> sorted_dataset["label"][:10]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
>>> sorted_dataset["label"][-10:]
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

在引擎盖下，这将创建一个索引列表，该列表根据列的值进行排序。然后，此索引映射用于访问基础Arrow表中的右侧行。

### Shuffle

shuffle()函数的作用是随机重新排列列值。您可以在此函数中指定生成器参数以使用不同的numpy.random。生成器，如果您希望对用于混洗数据集的算法进行更多控制。

```python
>>> shuffled_dataset = sorted_dataset.shuffle(seed=42)
>>> shuffled_dataset["label"][:10]
[1, 1, 1, 0, 1, 1, 1, 1, 1, 0]
```

Shuffling获取索引列表[0:len（my_dataset）]，并对其进行混洗以创建索引映射。然而，一旦数据集有了索引映射，速度就会变慢10倍。这是因为有一个额外的步骤可以使用索引映射来读取行索引，而且最重要的是，您不再读取连续的数据块。要恢复速度，您需要使用dataset.flatin_indices（）再次重写磁盘上的整个数据集，这将删除索引映射。或者，您可以切换到IterableDataset并利用其快速近似混洗IterableDataset.shuffle（）：

```bash
>>> iterable_dataset = dataset.to_iterable_dataset(num_shards=128)
>>> shuffled_iterable_dataset = iterable_dataset.shuffle(seed=42, buffer_size=1000)
```

### Select and Filter

有两个选项用于筛选数据集中的行：select（）和filter（）。

- select（）根据索引列表返回行：

```python
>>> small_dataset = dataset.select([0, 10, 20, 30, 40, 50])
>>> len(small_dataset)
6
```

- filter（）返回符合指定条件的行：

```python
>>> start_with_ar = dataset.filter(lambda example: example["sentence1"].startswith("Ar"))
>>> len(start_with_ar)
6
>>> start_with_ar["sentence1"]
['Around 0335 GMT , Tab shares were up 19 cents , or 4.4 % , at A $ 4.56 , having earlier set a record high of A $ 4.57 .',
'Arison said Mann may have been one of the pioneers of the world music movement and he had a deep love of Brazilian music .',
'Arts helped coach the youth on an eighth-grade football team at Lombardi Middle School in Green Bay .',
'Around 9 : 00 a.m. EDT ( 1300 GMT ) , the euro was at $ 1.1566 against the dollar , up 0.07 percent on the day .',
"Arguing that the case was an isolated example , Canada has threatened a trade backlash if Tokyo 's ban is not justified on scientific grounds .",
'Artists are worried the plan would harm those who need help most - performers who have a difficult time lining up shows .'
]
```

如果设置with_indices=True，filter（）也可以按索引进行筛选：

```python
>>> even_dataset = dataset.filter(lambda example, idx: idx % 2 == 0, with_indices=True)
>>> len(even_dataset)
1834
>>> len(dataset) / 2
1834.0
```

除非要保留的索引列表是连续的，否则这些方法还会在后台创建索引映射。

### Split

train_test_split（）函数可以创建训练和测试拆分（如果数据集还没有）。这允许您在每次拆分中调整相对比例或绝对采样数。在下面的示例中，使用test_size参数创建原始数据集的10%的测试拆分：

```python
>>> dataset.train_test_split(test_size=0.1)
{'train': Dataset(schema: {'sentence1': 'string', 'sentence2': 'string', 'label': 'int64', 'idx': 'int32'}, num_rows: 3301),
'test': Dataset(schema: {'sentence1': 'string', 'sentence2': 'string', 'label': 'int64', 'idx': 'int32'}, num_rows: 367)}
>>> 0.1 * len(dataset)
366.8
```

默认情况下，拆分是混洗的，但可以将shuffle设置为False以防止混洗。

### Shard

数据集支持分片，将非常大的数据集划分为预定义数量的块。在shard（）中指定num_shards参数，以确定要将数据集拆分为的碎片数。您还需要为要返回的shard提供index参数。

例如，imdb数据集有25000个示例：

```python
>>> from datasets import load_dataset
>>> datasets = load_dataset("imdb", split="train")
>>> print(dataset)
Dataset({
    features: ['text', 'label'],
    num_rows: 25000
})
```

将数据集分为四个块后，第一个分片将只有6250个示例：

```python
>>> dataset.shard(num_shards=4, index=0)
Dataset({
    features: ['text', 'label'],
    num_rows: 6250
})
>>>print(25000/4)
6250.0
```

## Rename, remove, cast, and flatten

以下函数允许您修改数据集的列。这些函数可用于重命名或删除列、将列更改为新的特征集以及展平嵌套的列结构。

### Rename

当需要重命名数据集中的列时，请使用rename_column（）。与原始列相关联的要素实际上会移动到新的列名下，而不仅仅是替换原来的列。
为rename_column（）提供原始列的名称和新列的名称：

```python
>>> dataset
Dataset({
    features: ['sentence1', 'sentence2', 'label', 'idx'],
    num_rows: 3668
})
>>> dataset = dataset.rename_column("sentence1", "sentenceA")
>>> dataset = dataset.rename_column("sentence2", "sentenceB")
>>> dataset
Dataset({
    features: ['sentenceA', 'sentenceB', 'label', 'idx'],
    num_rows: 3668
})
```

### Remove

当您需要删除一个或多个列时，请向remove_columns（）函数提供要删除的列名。通过提供列名列表删除多个列：

```python
>>> dataset = dataset.remove_columns("label")
>>> dataset
Dataset({
    features: ['sentence1', 'sentence2', 'idx'],
    num_rows: 3668
})
>>> dataset = dataset.remove_columns(["sentence1", "sentence2"])
>>> dataset
Dataset({
    features: ['idx'],
    num_rows: 3668
})
```

相反，select_columns（）选择一个或多个要保留的列，并删除其余列。此函数采用一个或一个列名列表：

```python
>>> dataset
Dataset({
    features: ['sentence1', 'sentence2', 'label', 'idx'],
    num_rows: 3668
})
>>> dataset = dataset.select_columns(['sentence1', 'sentence2', 'idx'])
>>> dataset
Dataset({
    features: ['sentence1', 'sentence2', 'idx'],
    num_rows: 3668
})
>>> dataset = dataset.select_columns('idx')
>>> dataset
Dataset({
    features: ['idx'],
    num_rows: 3668
})
```

### Cast

函数的作用是转换一列或多列的特征类型。此函数接受您的新功能作为其参数。下面的示例演示了如何更改ClassLabel和Value功能：

```python
>>> dataset.features
{'sentence1': Value(dtype='string', id=None),
'sentence2': Value(dtype='string', id=None),
'label': ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'], names_file=None, id=None),
'idx': Value(dtype='int32', id=None)}

>>> from datasets import ClassLabel, Value
>>> new_features = dataset.features.copy()
>>> new_features["label"] = ClassLabel(names=["negative", "positive"])
>>> new_features["idx"] = Value("int64")
>>> dataset = dataset.cast(new_features)
>>> dataset.features
{'sentence1': Value(dtype='string', id=None),
'sentence2': Value(dtype='string', id=None),
'label': ClassLabel(num_classes=2, names=['negative', 'positive'], names_file=None, id=None),
'idx': Value(dtype='int64', id=None)}
```

> 只有当原始特征类型和新特征类型兼容时，投射才有效。例如，如果原始列仅包含1和0，则可以将要素类型为Value（“int32”）的列强制转换为Value（”bool“）。

使用cast_column（）函数可以更改单个列的特征类型。将列名及其新功能类型作为参数传递：

```python
>>> dataset.features
{'audio': Audio(sampling_rate=44100, mono=True, id=None)}

>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
>>> dataset.features
{'audio': Audio(sampling_rate=16000, mono=True, id=None)}
```

### Flatten

有时，列可以是多种类型的嵌套结构。从SQuAD数据集中查看下面的嵌套结构：

```python
>>> from datasets import load_dataset
>>> dataset = load_dataset("squad", split="train")
>>> dataset.features
{'answers': Sequence(feature={'text': Value(dtype='string', id=None), 'answer_start': Value(dtype='int32', id=None)}, length=-1, id=None),
'context': Value(dtype='string', id=None),
'id': Value(dtype='string', id=None),
'question': Value(dtype='string', id=None),
'title': Value(dtype='string', id=None)}
```

答案字段包含两个子字段：text和answer_start。使用flatn（）函数将子字段提取到各自独立的列中：

```python
>>> flat_dataset = dataset.flatten()
>>> flat_dataset
Dataset({
    features: ['id', 'title', 'context', 'question', 'answers.text', 'answers.answer_start'],
 num_rows: 87599
})
```

请注意，子字段现在是它们自己独立的列：answers.text和answers.answer_start。

## Map

数据集的一些更强大的应用程序来自于使用map（）函数。map（）的主要目的是加快处理函数的速度。它允许您将处理函数独立地或分批地应用于数据集中的每个示例。此函数甚至可以创建新行和列。

在以下示例中，在数据集中的每个句子1值前面加上“我的句子：”。

首先创建一个函数，在每个句子的开头添加“我的句子：”。函数需要接受并输出dict：

```python
>>> def add_prefix(example):
...     example["sentence1"] = 'My sentence: ' + example["sentence1"]
...     return example
```

现在使用map（）将add_prefix函数应用于整个数据集：

```python
>>> updated_dataset = small_dataset.map(add_prefix)
>>> updated_dataset["sentence1"][:5]
['My sentence: Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
"My sentence: Yucaipa owned Dominick 's before selling the chain to Safeway in 1998 for $ 2.5 billion .",
'My sentence: They had published an advertisement on the Internet on June 10 , offering the cargo for sale , he added .',
'My sentence: Around 0335 GMT , Tab shares were up 19 cents , or 4.4 % , at A $ 4.56 , having earlier set a record high of A $ 4.57 .',
]
```

让我们来看另一个例子，除了这次，您将使用map（）删除一列。删除列时，只有在将示例提供给映射函数后，才会删除该列。这允许映射函数在删除列之前使用列的内容。

使用map（）中的remove_columns参数指定要删除的列：

```python
>>> updated_dataset = dataset.map(lambda example: {"new_sentence": example["sentence1"]}, remove_columns=["sentence1"])
>>> updated_dataset.column_names
['sentence2', 'label', 'idx', 'new_sentence']
```

> 数据集还有一个remove_columns（）函数，它的速度更快，因为它不会复制其余列的数据。

如果设置with_indices=True，也可以将map（）与索引一起使用。下面的例子将索引添加到每个句子的开头：

```python
>>> updated_dataset = dataset.map(lambda example, idx: {"sentence2": f"{idx}: " + example["sentence2"]}, with_indices=True)
>>> updated_dataset["sentence2"][:5]
['0: Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .',
 "1: Yucaipa bought Dominick 's in 1995 for $ 693 million and sold it to Safeway for $ 1.8 billion in 1998 .",
 "2: On June 10 , the ship 's owners had published an advertisement on the Internet , offering the explosives for sale .",
 '3: Tab shares jumped 20 cents , or 4.6 % , to set a record closing high at A $ 4.57 .',
 '4: PG & E Corp. shares jumped $ 1.63 or 8 percent to $ 21.03 on the New York Stock Exchange on Friday .'
]
```

### Multiprocessing

多处理器通过在CPU上并行处理大大加快了处理速度。在map（）中设置num_proc参数以设置要使用的进程数：

```python
updated_dataset = dataset.map(lambda example, idx: {"sentence2": f"{idx}: " + example["sentence2"]}, num_proc=4)
```

如果设置为_rank=True，map（）也适用于进程的秩。这类似于with_indices参数。映射函数中的with_rank参数位于索引1之后（如果它已经存在）。

```python
import torch
from multiprocess import set_start_method
from transformers import AutoTokenizer, AutoModelForCausalLM 
from datasets import load_dataset

# Get an example dataset
dataset = load_dataset("fka/awesome-chatgpt-prompts", split="train")

# Get an example model and its tokenizer 
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B-Chat").eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")

def gpu_computation(batch, rank):
    # Move the model on the right GPU if it's not there already
    device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
    model.to(device)
  
    # Your big GPU call goes here, for example:
    chats = [[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ] for prompt in batch["prompt"]]
    texts = [tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True
    ) for chat in chats]
    model_inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**model_inputs, max_new_tokens=512)
    batch["output"] = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return batch

if __name__ == "__main__":
    set_start_method("spawn")
    updated_dataset = dataset.map(
        gpu_computation,
        batched=True,
        batch_size=16,
        with_rank=True,
        num_proc=torch.cuda.device_count(),  # one process per GPU
    )
```

rank的主要用例是跨多个GPU并行计算。这需要设置multiprocess.set_start_method（“spawn”）。如果不这样做，您将收到以下CUDA错误：

```python
RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method.
```

### Batch processing

map（）函数支持处理一批批示例。通过设置batched=True对批次进行操作。默认的批大小是1000，但可以使用batch_size参数进行调整。批处理实现了有趣的应用程序，如将长句拆分为较短的块和数据扩充。

#### Split long examples

当示例太长时，您可能需要将它们分成几个较小的块。首先创建一个函数：

1. 将sentence1字段拆分为50个字符的块。
2. 将所有块堆叠在一起以创建新的数据集。

```python
def chunk_examples(examples):
    chunks = []
    for sentence in examples["sentence1"]:
        chunks += [sentence[i:i + 50] for i in range(0, len(sentence), 50)]
    return {"chunks": chunks}
```

使用map（）应用函数：

```python
>>> chunked_dataset = dataset.map(chunk_examples, batched=True, remove_columns=dataset.column_names)
>>> chunked_dataset[:10]
{'chunks': ['Amrozi accused his brother , whom he called " the ',
            'witness " , of deliberately distorting his evidenc',
            'e .',
            "Yucaipa owned Dominick 's before selling the chain",
            ' to Safeway in 1998 for $ 2.5 billion .',
            'They had published an advertisement on the Interne',
            't on June 10 , offering the cargo for sale , he ad',
            'ded .',
            'Around 0335 GMT , Tab shares were up 19 cents , or',
            ' 4.4 % , at A $ 4.56 , having earlier set a record']}
```

请注意，现在句子是如何被分割成较短的块的，并且数据集中有更多的行。

```python
>>> dataset
Dataset({
 features: ['sentence1', 'sentence2', 'label', 'idx'],
 num_rows: 3668
})
>>> chunked_dataset
Dataset({
    features: ['chunks'],
    num_rows: 10470
})
```

#### Data augmentation

map（）函数也可以用于数据扩充。以下示例为句子中的掩码标记生成附加单词。
在transformers's的FillMaskPipeline中加载并使用RoBERTA模型：

```python
from random import randint
from transformers import pipeline

fillmask = pipeline("fill-mask", model="roberta-base")
mask_token = fillmask.tokenizer.mask_token
smaller_dataset = dataset.filter(lambda e, i: i<100, with_indices=True)
```

创建一个函数，随机选择一个单词在句子中屏蔽。该函数还应返回原始句子和RoBERTA生成的前两个替换项。

```python
def augment_data(examples):
    outputs = []
    for sentence in examples["sentence1"]:
        words = sentence.split(' ')
        K = randint(1, len(words)-1)
        masked_sentence = " ".join(words[:K]  + [mask_token] + words[K+1:])
        predictions = fillmask(masked_sentence)
        augmented_sequences = [predictions[i]["sequence"] for i in range(3)]
        outputs += [sentence] + augmented_sequences

    return {"data": outputs}
```

使用map（）将函数应用于整个数据集：

```python
>>> augmented_dataset = smaller_dataset.map(augment_data, batched=True, remove_columns=dataset.column_names, batch_size=8)
>>> augmented_dataset[:9]["data"]
['Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
 'Amrozi accused his brother, whom he called " the witness ", of deliberately withholding his evidence.',
 'Amrozi accused his brother, whom he called " the witness ", of deliberately suppressing his evidence.',
 'Amrozi accused his brother, whom he called " the witness ", of deliberately destroying his evidence.',
 "Yucaipa owned Dominick 's before selling the chain to Safeway in 1998 for $ 2.5 billion .",
 'Yucaipa owned Dominick Stores before selling the chain to Safeway in 1998 for $ 2.5 billion.',
 "Yucaipa owned Dominick's before selling the chain to Safeway in 1998 for $ 2.5 billion.",
 'Yucaipa owned Dominick Pizza before selling the chain to Safeway in 1998 for $ 2.5 billion.'
]
```

对于每一个原始句子，RoBERTA用三个备选词扩充一个随机单词。原始的扭曲是由`withholding`, `suppressing`, and `destroying`所补充的。

### Process multiple splits

许多数据集具有可以使用DatasetDict.map（）同时处理的拆分。例如，将训练和测试中的sentence1字段标记为：

```python
>>> from datasets import load_dataset

# load all the splits
>>> dataset = load_dataset('glue', 'mrpc')
>>> encoded_dataset = dataset.map(lambda examples: tokenizer(examples["sentence1"]), batched=True)
>>> encoded_dataset["train"][0]
{'sentence1': 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
'sentence2': 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .',
'label': 1,
'idx': 0,
'input_ids': [  101,  7277,  2180,  5303,  4806,  1117,  1711,   117,  2292, 1119,  1270,   107,  1103,  7737,   107,   117,  1104,  9938, 4267, 12223, 21811,  1117,  2554,   119,   102],
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
```

### Distributed usage

在分布式设置中使用map（）时，还应使用torch.distributed.barrier。这确保主进程执行映射，而其他进程加载结果，从而避免重复工作。

以下示例显示了如何使用torch.distributed.barrier来同步进程：

```python
from datasets import Dataset
import torch.distributed

dataset1 = Dataset.from_dict({"a": [0, 1, 2]})

if training_args.local_rank > 0:
    print("Waiting for main process to perform the mapping")
    torch.distributed.barrier()

dataset2 = dataset1.map(lambda x: {"a": x["a"] + 1})

if training_args.local_rank == 0:
    print("Loading results from main process")
    torch.distributed.barrier()
```

## Concatenate

如果单独的数据集共享相同的列类型，则可以连接它们。使用Concatenate_dataset（）连接数据集：

```python
from datasets import concatenate_datasets, load_dataset

bookcorpus = load_dataset("bookcorpus", split="train")
wiki = load_dataset("wikipedia", "20220301.en", split="train")
wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])  # only keep the 'text' column

assert bookcorpus.features.type == wiki.features.type
bert_dataset = concatenate_datasets([bookcorpus, wiki])
```

您也可以通过设置axis=1水平连接两个数据集，只要数据集具有相同的行数：

```python
from datasets import Dataset
bookcorpus_ids = Dataset.from_dict({"ids": list(range(len(bookcorpus)))})
bookcorpus_with_ids = concatenate_datasets([bookcorpus, bookcorpus_ids], axis=1)
```

### Interleave

您还可以将多个数据集混合在一起，方法是从每个数据集中交替抽取示例来创建新的数据集。这被称为交错，由interleave_dataset（）函数启用。interleave_Dataset（）和concatenate_Dataset（）都适用于常规的Dataset和IterableDataset对象。有关如何交错IterableDataset对象的示例，请参阅流指南。

您可以定义每个原始数据集的采样概率，以指定如何交错数据集。在这种情况下，新的数据集是通过从随机数据集中逐个获取示例来构建的，直到其中一个数据集的样本用完为止。

```python
>>> seed = 42
>>> probabilities = [0.3, 0.5, 0.2]
>>> d1 = Dataset.from_dict({"a": [0, 1, 2]})
>>> d2 = Dataset.from_dict({"a": [10, 11, 12, 13]})
>>> d3 = Dataset.from_dict({"a": [20, 21, 22]})
>>> dataset = interleave_datasets([d1, d2, d3], probabilities=probabilities, seed=seed)
>>> dataset["a"]
[10, 11, 20, 12, 0, 21, 13]
```

您也可以指定stopping_strategy。默认策略first_exhausted是一种子采样策略，即一旦其中一个数据集的样本用完，就停止数据集构建。可以指定stopping_strategy=all_exhausted来执行过采样策略。在这种情况下，一旦每个数据集中的每个样本至少添加一次，数据集构建就会停止。在实践中，这意味着如果数据集用完，它将返回到该数据集的开头，直到达到停止标准。请注意，如果未指定采样概率，则新数据集将具有max_length_datasets*nb_dataset样本。

```python
>>> d1 = Dataset.from_dict({"a": [0, 1, 2]})
>>> d2 = Dataset.from_dict({"a": [10, 11, 12, 13]})
>>> d3 = Dataset.from_dict({"a": [20, 21, 22]})
>>> dataset = interleave_datasets([d1, d2, d3], stopping_strategy="all_exhausted")
>>> dataset["a"]
[0, 10, 20, 1, 11, 21, 2, 12, 22, 0, 13, 20]
```

## Format

函数的作用是更改列的格式，使其与一些常见的数据格式兼容。在类型参数和要格式化的列中指定所需的输出。格式化是动态应用的。

例如，通过设置type=“torch”来创建PyTorch张量：

```python
import torch
dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
```

with_format（）函数还更改列的格式，只是它返回一个新的数据集对象：

```python
dataset = dataset.with_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
```

> 数据集还支持其他常见的数据格式，如NumPy、Pandas和JAX。有关如何有效创建TensorFlow数据集的更多详细信息，请参阅《使用数据集与TensorFlow》指南。

如果需要将数据集重置为原始格式，请使用reset_format（）函数：

```python
>>> dataset.format
{'type': 'torch', 'format_kwargs': {}, 'columns': ['label'], 'output_all_columns': False}
>>> dataset.reset_format()
>>> dataset.format
{'type': 'python', 'format_kwargs': {}, 'columns': ['idx', 'label', 'sentence1', 'sentence2'], 'output_all_columns': False}
```

### Format transform

set_transform（）函数动态应用自定义格式转换。此函数可替换以前指定的任何格式。例如，您可以使用此函数来动态标记和填充标记。只有在访问示例时才应用标记化：

```python
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
>>> def encode(batch):
>>>     return tokenizer(batch["sentence1"], padding="longest", truncation=True, max_length=512, return_tensors="pt")
>>> dataset.set_transform(encode)
>>> dataset.format
{'type': 'custom', 'format_kwargs': {'transform': <function __main__.encode(batch)>}, 'columns': ['idx', 'label', 'sentence1', 'sentence2'], 'output_all_columns': False}
```

您也可以使用set_transform（）函数来解码Features不支持的格式。例如，音频功能使用soundfile——一个快速简单的安装库——但它不支持不太常见的音频格式。这里是您可以使用set_transform（）动态应用自定义解码转换的地方。您可以自由使用任何您喜欢的库来解码音频文件。

以下示例使用pydub包打开声音文件不支持的音频格式：

```python
import numpy as np
from pydub import AudioSegment

audio_dataset_amr = Dataset.from_dict({"audio": ["audio_samples/audio.amr"]})

def decode_audio_with_pydub(batch, sampling_rate=16_000):
    def pydub_decode_file(audio_path):
        sound = AudioSegment.from_file(audio_path)
        if sound.frame_rate != sampling_rate:
            sound = sound.set_frame_rate(sampling_rate)
        channel_sounds = sound.split_to_mono()
        samples = [s.get_array_of_samples() for s in channel_sounds]
        fp_arr = np.array(samples).T.astype(np.float32)
        fp_arr /= np.iinfo(samples[0].typecode).max
        return fp_arr

    batch["audio"] = [pydub_decode_file(audio_path) for audio_path in batch["audio"]]
    return batch

audio_dataset_amr.set_transform(decode_audio_with_pydub)
```

## Save

处理完数据集后，可以使用save_to_disk（）保存并在以后重用它。
通过提供要保存数据集的目录的路径来保存数据集：

```python
encoded_dataset.save_to_disk("path/of/my/dataset/directory")
```

使用load_from_disk（）函数重新加载数据集：

```python
from datasets import load_from_disk
reloaded_dataset = load_from_disk("path/of/my/dataset/directory")
```

> 想将数据集保存到云存储提供商吗？阅读我们的云存储指南，了解如何将数据集保存到AWS或谷歌云存储。

## Export

数据集也支持导出，因此您可以在其他应用程序中使用数据集。下表显示了当前支持的可以导出为的文件格式：


| File type | Export method                                                                                                                       |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| CSV       | [Dataset.to\_csv()](https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/main_classes#datasets.Dataset.to_csv)         |
| JSON      | [Dataset.to\_json()](https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/main_classes#datasets.Dataset.to_json)       |
| Parquet   | [Dataset.to\_parquet()](https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/main_classes#datasets.Dataset.to_parquet) |
| SQL       | [Dataset.to\_sql()](https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/main_classes#datasets.Dataset.to_sql)         |

内存中Python对象Dataset.to_pandas（）或Dataset.to_dict（）

例如，将数据集导出到CSV文件，如下所示：

```python
encoded_dataset.to_csv("path/of/my/dataset.csv")
```

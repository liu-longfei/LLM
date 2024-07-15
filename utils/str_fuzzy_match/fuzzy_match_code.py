# 字符串分句模糊匹配

import spacy
from rapidfuzz import process


text_content = """
    从前有一只小猫，名叫咪咪。咪咪非常喜欢探索世界。
    有一天，它发现了一座神秘的花园。花园里开满了五颜六色的花朵，还有一个闪闪发光的小池塘。
咪咪在花园里结识了一只友好的小鸟，他们每天一起玩耍，分享美味的果实和有趣的故事。

    咪咪感到非常幸福，因为它找到了一个充满惊喜和快乐的新家。
"""

model_content = """
花园里开满了五....颜六色的花朵，还有一个闪闪发----光的小池塘。
 咪咪感到非{}常幸福，因为它找到了一个充[]满惊喜和快乐的新家
"""

# 分句
nlp = spacy.load("zh_core_web_sm")
sentences = [sent.text for sent in nlp(text_content).sents]
print(sentences)

result = process.extract(model_content, sentences)
print(result)

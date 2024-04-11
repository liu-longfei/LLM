import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 关于word embedding,以序列建模为例
# 考虑source sentence 和 target sentence
# 构建序列，序列的字符以其词表中的索引的形式表示
batch_size = 2

# 单词表大小
max_num_src_words = 8
max_num_tgt_words = 8
model_dim = 8

# 序列的最大长度
max_src_seq_len = 5
max_tgt_seq_len = 5
max_position_len = 5

src_len = torch.Tensor([2, 4]).to(torch.int32)
tgt_len = torch.Tensor([4, 3]).to(torch.int32)

# 单词索引构成的源句子和目标句子，构建batch，并且做了padding，默认值为0
src_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1, max_num_src_words, (L,)), (0, max(src_len)-L)), 0) for L in src_len])
tgt_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1, max_num_src_words, (L,)), (0, max(src_len)-L)), 0) for L in tgt_len])

# 构造embedding
src_embedding_table = nn.Embedding(max_num_src_words+1, model_dim)
tgt_embedding_table = nn.Embedding(max_num_tgt_words+1, model_dim)
src_embedding = src_embedding_table(src_seq)
tgt_embedding = tgt_embedding_table(tgt_seq)

# 构造postion embedding
pos_mat = torch.arange(max_position_len).reshape((-1, 1))
i_mat = torch.pow(1000, torch.arange(0, 8, 2).reshape((1, -1))/model_dim)
pe_embedding_table = torch.zeros(max_position_len, model_dim)
pe_embedding_table[:, 0::2] = torch.sin(pos_mat / i_mat)
pe_embedding_table[:, 1::2] = torch.cos(pos_mat / i_mat)
# print(pe_embedding_table)

pe_embedding = nn.Embedding(max_position_len, model_dim)
pe_embedding.weight = nn.Parameter(pe_embedding_table, requires_grad=False)
# print(pe_embedding.weight)

src_pos = torch.cat([torch.unsqueeze(torch.arange(max(src_len)), 0) for _ in src_len]).to(torch.int32)
tgt_pos = torch.cat([torch.unsqueeze(torch.arange(max(tgt_len)), 0) for _ in tgt_len]).to(torch.int32)

src_pe_embedding = pe_embedding(src_pos)
tgt_pe_embedding = pe_embedding(tgt_pos)

# 构造encoder的self-attention mask
# mask的shape: [batch_size, max_src_len, max_src_len],值为1或-inf
valid_encoder_pos = torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max(src_len)-L)), 0) for L in src_len]), 2)
valid_encoder_pos_matrix = torch.bmm(valid_encoder_pos, valid_encoder_pos.transpose(1, 2))
invalid_encoder_pos_matrix = 1 - valid_encoder_pos_matrix
mask_encoder_self_attention = invalid_encoder_pos_matrix.to(torch.bool)

score = torch.randn(batch_size, max(src_len), max(src_len))
masked_score = score.masked_fill(mask_encoder_self_attention, -1e9)
prob = F.softmax(masked_score, -1)

# print(src_len)
# print(score)
# print(masked_score)
# print(prob)


# Step 5:构造intra-attention的mask
# Q @ K^T shape: [batch_size, tgt_seq_len, src_seq_len]
valid_encoder_pos = torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max(src_len)-L)), 0) for L in src_len]), 2)
valid_decoder_pos = torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max(src_len)-L)), 0) for L in tgt_len]), 2)
valid_cross_pos_matrix = torch.bmm(valid_decoder_pos, valid_encoder_pos.transpose(1, 2))
invalid_cross_pos_matrix = 1 - valid_cross_pos_matrix
mask_cross_attention = invalid_cross_pos_matrix.to(torch.bool)

# Step 6: 构造decoder self-attention的mask
valid_decoder_tri_matrix = torch.cat([torch.unsqueeze(F.pad(torch.tril(torch.ones((L, L))), (0, max(tgt_len)-L, 0, max(tgt_len)-L)), 0) for L in tgt_len])
invalid_decoder_tri_matrix = 1 - valid_decoder_tri_matrix
invalid_decoder_tri_matrix = invalid_decoder_tri_matrix.to(torch.bool)
score = torch.randn(batch_size, max(tgt_len), max(tgt_len))
masked_score = score.masked_fill(invalid_decoder_tri_matrix, -1e9)
prob = F.softmax(masked_score, -1)
print(tgt_len)
print(prob)


# Step 7: 构建scaled self-attention
def scaled_dot_product_attention(Q, K, V, attn_mask):
    # shape of Q, K, V : (batch_size*num_head, seq_len, model_dim/num_head)
    score = torch.bmm(Q, K.transpose(-2, -1))/torch.sqrt(model_dim)
    masked_score = score.masked_fill(attn_mask, -1e9)
    prob = F.softmax(masked_score, -1)
    context = torch.bmm(prob, V)
    return context




# # softmax演示，scaled的重要性
# alpha1 = 0.1
# alpha2 = 10
# score = torch.randn(5)
# prob1 = F.softmax(score*alpha1, -1)
# prob2 = F.softmax(score*alpha2, -1)
# def softmax_func(score):
#     return F.softmax(score)
# jaco_mat1 = torch.autograd.functional.jacobian(softmax_func, score*alpha1)
# jaco_mat2 = torch.autograd.functional.jacobian(softmax_func, score*alpha2)
# print(jaco_mat1)
# print(jaco_mat2)

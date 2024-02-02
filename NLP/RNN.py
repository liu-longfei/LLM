import torch
import torch.nn as nn


bs, T = 2, 3  # 批大小， 输入序列长度
input_size, hidden_size = 2, 3  # 输入特征大小，隐含层特征大小
input = torch.randn(bs, T, input_size)  # 随机初始化一个特征序列
h_prev = torch.zeros(bs, hidden_size)  # 初始隐含状态

# 第一步：调用pytorch的rnn api
rnn = nn.RNN(input_size, hidden_size, batch_first=True)
rnn_output, state_final = rnn(input, h_prev.unsqueeze(0))
print('API:')
print(rnn_output)
print(state_final)


# 第二步：手写一个rnn_forward函数，实现单向RNN的计算原理
def rnn_forward(input, weight_ih, weight_hh, bias_ih, bias_hh, h_prev):
    bs, T, input_size = input.shape
    h_dim = weight_ih.shape[0]
    h_out = torch.zeros(bs, T, h_dim)  # 初始化一个输出（状态）矩阵

    for t in range(T):
        x = input[:, t, :].unsqueeze(2)  # 获取当前时刻输入特征，bs*input_size*1
        w_ih_batch = weight_ih.unsqueeze(0).tile(bs, 1, 1)  # bs*h_dim*input_size
        w_hh_batch = weight_hh.unsqueeze(0).tile(bs, 1, 1)  # bs*h_dim*h_dim

        w_times_x = torch.bmm(w_ih_batch, x).squeeze(-1)  # bs*h_dim
        w_times_h = torch.bmm(w_hh_batch, h_prev.unsqueeze(2)).squeeze(-1)  # bs*h_dim
        # print(w_times_x.shape, bias_ih.shape, w_times_h.shape, bias_hh.shape)
        h_prev = torch.tanh(w_times_x + bias_ih + w_times_h + bias_hh)

        h_out[:, t, :] = h_prev

    return h_out, h_prev.unsqueeze(0)


# for k, v in rnn.named_parameters():
#     print(k, '|', v)

custom_rnn_output, custom_state_final = rnn_forward(input, rnn.weight_ih_l0, rnn.weight_hh_l0, rnn.bias_ih_l0,
                                                    rnn.bias_hh_l0, h_prev)
print('rnn_forward:')
print(custom_rnn_output)
print(custom_state_final)

print('----------------------------------------------')


# 第三步：实现一个bidirectional_rnn_forward函数，实现双向RNN的计算原理
def bidirectional_rnn_forward(input, weight_ih, weight_hh, bias_ih, bias_hh, h_prev,
                              weight_ih_reverse, weight_hh_reverse, bias_ih_reverse, bias_hh_reverse, h_prev_reverse):
    bs, T, input_size = input.shape
    h_dim = weight_ih.shape[0]
    h_out = torch.zeros(bs, T, h_dim * 2)  # 初始化一个输出（状态）矩阵

    forward_output = rnn_forward(input, weight_ih, weight_hh, bias_ih, bias_hh, h_prev)[0]
    backward_output = \
    rnn_forward(torch.flip(input, [1]), weight_ih_reverse, weight_hh_reverse, bias_ih_reverse, bias_hh_reverse,
                h_prev_reverse)[0]

    h_out[:, :, :h_dim] = forward_output
    h_out[:, :, h_dim:] = backward_output

    return h_out, h_out[:, -1, :].reshape((bs, 2, h_dim)).transpose(0, 1)


# 验证一下bidirectional_rnn_forward的正确性
bi_rnn = nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=True)
h_prev = torch.zeros(2, bs, hidden_size)
bi_rnn_output, bi_state_final = bi_rnn(input, h_prev)

print('bi_RNN API:')
print(bi_rnn_output)
print(bi_state_final)

# for k, v in bi_rnn.named_parameters():
#     print(k, '|', v)

custom_bi_rnn_output, custom_bi_state_final = bidirectional_rnn_forward(input, bi_rnn.weight_ih_l0, bi_rnn.weight_hh_l0,
                                                                        bi_rnn.bias_ih_l0, bi_rnn.bias_hh_l0, h_prev[0],
                                                                        bi_rnn.weight_ih_l0_reverse,
                                                                        bi_rnn.weight_hh_l0_reverse,
                                                                        bi_rnn.bias_ih_l0_reverse,
                                                                        bi_rnn.bias_hh_l0_reverse, h_prev[1])

print('bi_rnn_forward:')
print(custom_bi_rnn_output)
print(custom_bi_state_final)


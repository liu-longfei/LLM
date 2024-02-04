
import torch
import torch.nn as nn


# 实现LSTM和LSTMP源码
# 定义常量
bs, T, i_size, h_size = 2, 3, 4, 5
# proj_size
input = torch.randn(bs, T, i_size)  # 输入序列
c0 = torch.randn(bs, h_size)  # 初始值，不需要训练
h0 = torch.randn(bs, h_size)

# 调用官方LSTM API
lstm_layer = nn.LSTM(i_size, h_size, batch_first=True)
output, (h_final, c_final) = lstm_layer(input, (h0.unsqueeze(0), c0.unsqueeze(0)))
print(output)

# for k, v in lstm_layer.named_parameters():
#     print(k, '|', v.shape)


# 自己写一个LSTM模型
def lstm_forward(input, initial_states, w_ih, w_hh, b_ih, b_hh):
    h0, c0 = initial_states  # 初始状态
    bs, T, i_size = input.shape
    h_size = w_ih.shape[0] // 4

    prev_h = h0
    prev_c = c0
    batch_w_ih = w_ih.unsqueeze(0).tile(bs, 1, 1)  # [bs, 4*h_size, i_size]
    batch_w_hh = w_hh.unsqueeze(0).tile(bs, 1, 1)  # [bs, 4*h_size, h_size]

    output_size = h_size
    output = torch.zeros(bs, T, output_size)  # 输出序列

    for t in range(T):
        x = input[:, t, :]  # 当前时刻的输入向量，[bs, i_size]
        w_times_x = torch.bmm(batch_w_ih, x.unsqueeze(-1))  # [bs, 4*h_size, 1]
        w_times_x = w_times_x.squeeze(-1)

        w_times_h_prev = torch.bmm(batch_w_hh, prev_h.unsqueeze(-1))  # [bs, 4*h_size, 1]
        w_times_h_prev = w_times_h_prev.squeeze(-1)

        # 分别计算输入门（i）、遗忘门（f）、cell门（g）、输出门（o）
        i_t = torch.sigmoid(w_times_x[:, :h_size] + w_times_h_prev[:, :h_size] + b_ih[:h_size] + b_hh[:h_size])
        f_t = torch.sigmoid(w_times_x[:, h_size:2*h_size] + w_times_h_prev[:, h_size:2*h_size]
                            + b_ih[h_size:2*h_size] + b_hh[h_size:2*h_size])
        g_t = torch.tanh(w_times_x[:, 2*h_size:3*h_size] + w_times_h_prev[:, 2*h_size:3*h_size]
                            + b_ih[2*h_size:3*h_size] + b_hh[2*h_size:3*h_size])
        o_t = torch.sigmoid(w_times_x[:, 3*h_size:4*h_size] + w_times_h_prev[:, 3*h_size:4*h_size]
                            + b_ih[3*h_size:4*h_size] + b_hh[3*h_size:4*h_size])
        prev_c = f_t * prev_c + i_t * g_t
        prev_h = o_t * torch.tanh(prev_c)

        output[:, t, :] = prev_h

    return output, (prev_h, prev_c)


output_custom, (h_final_custom, c_final_custom) = lstm_forward(input, (h0, c0), lstm_layer.weight_ih_l0,
                                                               lstm_layer.weight_hh_l0, lstm_layer.bias_ih_l0,
                                                               lstm_layer.bias_hh_l0)

print(output_custom)

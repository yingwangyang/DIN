import torch
import torch.nn as nn
import torch.nn.functional as F

from fc import FCLayer


class DinAttentionLayer(nn.Module):
    def __init__(self, embedding_dim=36):
        super(DinAttentionLayer, self).__init__()

        self.local_att = LocalActivationUnit(hidden_size=[80, 40, 1],
                                             bias=[True, True, True],
                                             embedding_dim=embedding_dim,
                                             batch_norm=False)
        
    def forward(self, query_ad, user_behavior, user_behavior_length):
        # query ad             : batch_size * embedding_size
        # user_behavior        : batch_size * time_seq_len * embedding_size
        # user_behavior_length : batch_size * time_seq_len
        # output               : batch_size * 1 * embedding_size

        attention_score = self.local_att(query_ad, user_behavior)  # batch_size * time_seq_len * 1
        attention_score = torch.transpose(attention_score, 1, 2)   # batch_size * 1 * time_seq_len

        # define mask by length
        user_behavior_length = user_behavior_length.long()
        mask = torch.arange(
            user_behavior.size(1), device=user_behavior.device
        )[None, :] < user_behavior_length[:, None]  # batch_size * time_seq_len

        # mask
        score = attention_score * mask.float()  # batch_size * 1 * time_seq_len
        score = F.softmax(score, dim=-1)                                    # batch_size * 1 * time_seq_len

        # multiply weighted sum
        output = torch.matmul(score, user_behavior) # batch_size * 1 * embedding_size
        return output  # batch_size * 1 * embedding_size


class LocalActivationUnit(nn.Module):
    def __init__(self, hidden_size=[80, 40, 1], bias=[True, True, True], embedding_dim=4, batch_norm=False):
        super(LocalActivationUnit, self).__init__()

        self.fc1 = FCLayer(input_size=4*embedding_dim,
                           hidden_size=hidden_size[0],
                           bias=bias[0],
                           batch_norm=batch_norm,
                           activation='none',
                           use_sigmoid=True)
        self.fc2 = FCLayer(input_size=hidden_size[0],
                           hidden_size=hidden_size[1],
                           bias=bias[1],
                           batch_norm=batch_norm,
                           activation='none',
                           use_sigmoid=True)
        self.fc3 = FCLayer(input_size=hidden_size[1],
                           hidden_size=hidden_size[2],
                           bias=bias[2],
                           batch_norm=batch_norm,
                           activation='none')
        
    def forward(self, query, user_behavior):
        # query ad      : size -> batch_size * embedding_size
        # user_behavior : size -> batch_size * time_seq_len * embedding_size

        user_behavior_len = user_behavior.size(1)   # time_seq_len
        queries = query.unsqueeze(1).repeat(1, user_behavior_len, 1)

        attention_input = torch.cat([queries, user_behavior, queries - user_behavior, queries * user_behavior], dim=-1) # batch_size * time_seq_len * (4*embedding_size)
        attention_out = self.fc3(self.fc2(self.fc1(attention_input)))  # batch_size * time_seq_len * 1

        return attention_out  # batch_size * time_seq_len * 1

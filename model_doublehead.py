import torch
import torch.nn as nn

from attention import DinAttentionLayer
from embedding import EmbeddingLayer
from fc import FCLayer


class DoubleHeadDIN(nn.Module):
    def __init__(self, n_uid, n_mid, n_cat, embedding_dim=12, hidden_dim=None, dropout_rate=0.2):
        super(DoubleHeadDIN, self).__init__()
        if hidden_dim is None:
            hidden_dim = [108, 200, 80]

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.uid_embeddings = EmbeddingLayer(n_uid, self.embedding_dim)
        self.mid_embeddings = EmbeddingLayer(n_mid, self.embedding_dim)
        self.cat_embeddings = EmbeddingLayer(n_cat, self.embedding_dim)
        self.attn = DinAttentionLayer(embedding_dim=self.embedding_dim * 2)

        mlp_input_dim = self.embedding_dim * 9
        self.shared_mlp = nn.Sequential(
            FCLayer(mlp_input_dim, hidden_size=self.hidden_dim[1], bias=True, batch_norm=True, activation="dice"),
            nn.Dropout(p=dropout_rate),
            FCLayer(self.hidden_dim[1], hidden_size=self.hidden_dim[2], bias=True, activation="dice"),
            nn.Dropout(p=dropout_rate),
        )
        self.preference_head = nn.Linear(self.hidden_dim[2], 1)
        self.hesitation_head = nn.Linear(self.hidden_dim[2], 1)

        for head in [self.preference_head, self.hesitation_head]:
            nn.init.xavier_normal_(head.weight, gain=1.0)
            nn.init.zeros_(head.bias)

    def forward(self, uids, mids, cats, mid_his, cat_his, mid_mask):
        uid_batch_eb = self.uid_embeddings(uids)
        mid_batch_eb = self.mid_embeddings(mids)
        cat_batch_eb = self.cat_embeddings(cats)
        mid_his_batch_eb = self.mid_embeddings(mid_his)
        cat_his_batch_eb = self.cat_embeddings(cat_his)

        item_eb = torch.concat((mid_batch_eb, cat_batch_eb), dim=1)
        item_his_eb = torch.concat((mid_his_batch_eb, cat_his_batch_eb), dim=2)
        item_his_eb_sum = torch.sum(item_his_eb, dim=1)

        attention_output = self.attn(item_eb, item_his_eb, mid_mask)
        att_fea = torch.sum(attention_output, dim=1)
        inp = torch.concat((uid_batch_eb, item_eb, item_his_eb_sum, item_eb * item_his_eb_sum, att_fea), dim=-1)

        shared = self.shared_mlp(inp)
        pref_logit = self.preference_head(shared).squeeze(-1)
        hes_logit = self.hesitation_head(shared).squeeze(-1)
        return pref_logit, hes_logit

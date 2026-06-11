import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding import EmbeddingLayer
from fc import FCLayer
from attention import DinAttentionLayer


class DeepInterestNetwork(nn.Module):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_DIM=[162, 200, 80, 2]):
        super(DeepInterestNetwork, self).__init__()
        self.embedding_dim = EMBEDDING_DIM
        self.hid_dim = HIDDEN_DIM

        # Embedding layers
        self.uid_embeddings = EmbeddingLayer(n_uid, self.embedding_dim)
        self.mid_embeddings = EmbeddingLayer(n_mid, self.embedding_dim)
        self.cat_embeddings = EmbeddingLayer(n_cat, self.embedding_dim)

        self.attn = DinAttentionLayer(embedding_dim=self.embedding_dim*2)
        mlp_input_dim = self.embedding_dim * 9  # 3 embeddings * 2 (ad + attn) + 3 features
        self.mlp = nn.Sequential(
            FCLayer(mlp_input_dim, hidden_size=self.hid_dim[1], bias=True, batch_norm=True, activation='dice'),
            FCLayer(self.hid_dim[1], hidden_size=self.hid_dim[2], bias=True, activation='dice'),
            FCLayer(self.hid_dim[2], hidden_size=self.hid_dim[3], bias=False, activation='none') 
        )
        uid_params = sum(p.numel() for p in self.uid_embeddings.parameters() if p.requires_grad)
        print(f"UID Embedding trainable params: {uid_params}")
        mid_params = sum(p.numel() for p in self.mid_embeddings.parameters() if p.requires_grad)
        print(f"MID Embedding trainable params: {mid_params}")
        cat_params = sum(p.numel() for p in self.cat_embeddings.parameters() if p.requires_grad)
        print(f"CAT Embedding trainable params: {cat_params}")
        att_params = sum(p.numel() for p in self.attn.parameters() if p.requires_grad)
        print(f"Attention Layer trainable params: {att_params}")
        mlp_params = sum(p.numel() for p in self.mlp.parameters() if p.requires_grad)
        print(f"MLP Layer trainable params: {mlp_params}")

    def forward(self, uids, mids, cats, mid_his, cat_his, mid_mask, noclk_mids=None, noclk_cats=None, use_negsampling=False):
        """input: uids, mids, cats, mid_his, cat_his, mid_mask, noclk_mids, noclk_cats
        """
        # item_eb, item_his_eb, mask
        uid_batch_eb = self.uid_embeddings(uids) # [B, emb_dim]
        mid_batch_eb = self.mid_embeddings(mids)
        cat_batch_eb = self.cat_embeddings(cats)
        mid_his_batch_eb = self.mid_embeddings(mid_his) # [128, 100, 18]
        cat_his_batch_eb = self.cat_embeddings(cat_his)
        
        item_eb = torch.concat((mid_batch_eb, cat_batch_eb), 1) # [128, 36]
        item_his_eb = torch.concat((mid_his_batch_eb, cat_his_batch_eb), 2) # [128, 100, 36]
        item_his_eb_sum = torch.sum(item_his_eb, dim=1) # [128, 36]
        
        if use_negsampling:
            if noclk_mids is None or noclk_cats is None:
                raise ValueError("Negative sampling tensors are required when use_negsampling=True.")
            noclk_mid_his_batch_eb = self.mid_embeddings(noclk_mids)
            noclk_cat_his_batch_eb = self.cat_embeddings(noclk_cats)
            noclk_item_his_eb = torch.concat((noclk_mid_his_batch_eb[:, :, 0, :], noclk_cat_his_batch_eb[:, :, 0, :]), -1)
            noclk_item_his_eb = noclk_item_his_eb.reshape(-1, noclk_mid_his_batch_eb.shape[1], 36)
            noclk_his_eb = torch.concat((noclk_mid_his_batch_eb, noclk_cat_his_batch_eb), -1)
            noclk_his_eb_sum_1 = torch.sum(noclk_his_eb, dim=2)
            noclk_his_eb_sum = torch.sum(noclk_his_eb_sum_1, 1)
        
        attention_output = self.attn(item_eb, item_his_eb, mid_mask) # [128, 1, 36]
        att_fea = torch.sum(attention_output, dim=1)
        inp = torch.concat((uid_batch_eb, item_eb, item_his_eb_sum, item_eb * item_his_eb_sum, att_fea), dim=-1) # [128, 162]

        y_hat = F.softmax(self.mlp(inp), dim=-1)

        return y_hat
    

if __name__ == "__main__":
    B = 128
    sl = 100
    # Define vocab sizes based on dataset stats
    n_uid = 543060
    n_mid = 367983
    n_cat = 1601

    # Generate integer index tensors for embeddings
    uids = torch.randint(0, n_uid, (B,), dtype=torch.long)
    mids = torch.randint(0, n_mid, (B,), dtype=torch.long)
    cats = torch.randint(0, n_cat, (B,), dtype=torch.long)

    # Generate user behavior histories as indices
    mid_his = torch.randint(0, n_mid, (B, sl), dtype=torch.long)
    cat_his = torch.randint(0, n_cat, (B, sl), dtype=torch.long)

    # Attention mask expects lengths per batch item (shape [B])
    mid_mask = torch.randint(1, sl + 1, (B,), dtype=torch.long)

    # Negative sampling candidates (indices)
    noclk_mids = torch.randint(0, n_mid, (B, sl, 5), dtype=torch.long)
    noclk_cats = torch.randint(0, n_cat, (B, sl, 5), dtype=torch.long)

    model = DeepInterestNetwork(n_uid=n_uid, n_mid=n_mid, n_cat=n_cat, EMBEDDING_DIM=12)

    y = model(uids, mids, cats, mid_his, cat_his, mid_mask, noclk_mids, noclk_cats)
    print(y.shape)

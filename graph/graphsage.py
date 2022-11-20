import torch
import torch.nn as nn
import math


class GraphSage(nn.Module):
    def __init__(self, aggregator, k, in_ft, hid_ft):
        super(GraphSage, self).__init__()
        self.aggregator = aggregator
        self.k = k
        self.activation = nn.ReLU()
        self.fc = nn.Linear(in_ft, hid_ft)
        self.w_k = nn.ModuleList([nn.Linear(2 * hid_ft, hid_ft) for i in range(k)])
        self.w_key = nn.ModuleList([nn.Linear(hid_ft, hid_ft) for i in range(k)])
        self.w_query = nn.ModuleList([nn.Linear(hid_ft, hid_ft) for i in range(k)])
        self.w_value = nn.ModuleList([nn.Linear(hid_ft, hid_ft) for i in range(k)])

    def forward(self, h, G):
        """
        G: [batch,n_sta,n_sta]
        """
        h = self.activation(self.fc(h))
        for i in range(self.k):
            attn_score = self.aggregator(self.w_key[i](h), self.w_query[i](h), G)
            h_kn = torch.bmm(attn_score, self.w_value[i](h))
            h_ = self.w_k[i](torch.concat((h, h_kn), dim=-1))
            h = self.activation(h_)
            # norm2 h
        return h

    def inductive(self, H, l, G):
        """
        h: [batch_size,n_stas,n_fts]
        l: [batch_size,n_stas]
        G: [batch size,n_stas,n_stas]
        """
        # breakpoint()
        list_h = []
        for _ in range(H.shape[1]):
            h = H[:, _, :, :]
            knn_station = torch.argsort(l.squeeze(), -1)[:, -3:]
            mask = torch.zeros_like(l, dtype=torch.int32)
            for j in range(l.shape[0]):
                for i in knn_station[j]:
                    mask[j, i] = 1

            h = self.activation(self.fc(h))
            idw_vector = torch.bmm(l.unsqueeze(1), h)
            h_x = idw_vector
            for i in range(self.k):
                attn_score = self.aggregator(self.w_key[i](h), self.w_query[i](h), G)
                h_kn = torch.bmm(attn_score, self.w_value[i](h))
                h_ = self.w_k[i](torch.concat((h, h_kn), dim=-1))

                h_x_attn_score = self.aggregator(self.w_key[i](h), self.w_query[i](h_x), mask.unsqueeze(1))
                h_kn_x = torch.bmm(h_x_attn_score, self.w_value[i](h))
                h_x = self.activation(self.w_k[i](torch.concat((h_x, h_kn_x), dim=-1)))

                h = self.activation(h_)
            list_h.append(h_x.squeeze())
        # norm 2 h,h_x
        return torch.stack(list_h, 1)


class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, key, query, mask=None):
        """_summary_
        Args:
            key (_type_): tensor([1,n_station,d_dim])
            query (_type_): tensor([1,n_station,d_dim])
            mask (_type_, optional): _description_. Defaults to None.
        Returns:
            _type_: _description_
        """
        n_dim = key.shape[-1]
        # breakpoint()
        score = torch.bmm(query, key.transpose(1, 2)) / torch.sqrt(
            torch.tensor([n_dim], device=query.device)
        )
        # breakpoint()
        if mask is not None:
            # mask = mask.squeeze()
            score = score.masked_fill(mask == 0, -math.inf)
        # attn = self.softmax(score.view(-1, n_station))
        return self.softmax(score)


if __name__ == '__main__':
    aggr = DotProductAttention()
    graph = GraphSage(aggr, 2, 128, 128)
    x = torch.rand(12, 6, 128)
    g = torch.rand(12, 6, 6)
    l = torch.rand(12, 6)
    h = graph.inductive(x, l, g)
    breakpoint()

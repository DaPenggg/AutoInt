# coding=utf-8

import torch
import torch.nn as NN
import torch.nn.functional as F


# AUTOINT: AUTOMATIC FEATURE INTERACTION LEARNING
class AutoIN(NN.Module):
    """'auto In model."""

    def __init__(self, cal_size, values_size, embeding_size, cal_num=1,value_num=2,dropout=0.5):
        '''
        :param cal_size: 类别属性的总个数：比如颜色：4个，年龄：4个，那么设置为8。
        :param values_size: 数值型的总个数
        :param embeding_size: 嵌入层的维度
        :param cal_num: 类别个数，比如只有颜色和年龄，那么设置为2。
        :param value_num:数值型的类型个数
        :param dropout: dropout
        '''
        super(AutoIN, self).__init__()
        self.cal_embedinds = NN.Embedding(num_embeddings=cal_size,
                                          embedding_dim=embeding_size,
                                          _weight=NN.init.xavier_normal_(torch.empty(cal_size,embeding_size),
                                                                         gain=NN.init.calculate_gain('relu')))
        self.value_embedings = NN.Embedding(num_embeddings=values_size,
                                            embedding_dim=embeding_size,
                                            _weight=NN.init.xavier_normal_(torch.empty(values_size,embeding_size),
                                                                           gain=NN.init.calculate_gain('relu')))
        self.MultiHead_1 = MultiHead(model_dim=embeding_size, output_dim=embeding_size // 2, num_head=8, dropout=0.5)
        self.MultiHead_2 = MultiHead(model_dim=embeding_size // 2, output_dim=embeding_size // 2, num_head=4,
                                     dropout=0.5)
        self.MultiHead_3 = MultiHead(model_dim=embeding_size // 2, output_dim=embeding_size // 2, num_head=4,
                                     dropout=0.5)
        num_output = value_num+cal_num
        self.fc_final = NN.Linear(in_features=num_output*embeding_size // 2, out_features=1, bias=False)
        self.dropout = dropout

        self.values_size = values_size
        self.embeding_size = embeding_size
        self.cal_size = cal_size

    def forward(self, cal_index, value_data, value_index):
        batch_size = value_data.size(0)
        cal_dim = self.cal_embedinds(cal_index)
        value_dim = self.value_embedings(value_index)
        value_dim = value_data*value_dim
        data_dim = torch.cat([value_dim, cal_dim], 1)

        # attention base on transformer structure. See NLP BERT module.
        print(data_dim.size())
        data_dim = self.MultiHead_1(data_dim)
        data_dim = F.relu(data_dim)
        print(data_dim.size())
        data_dim = self.MultiHead_2(data_dim)
        data_dim = F.relu(data_dim)
        print(data_dim.size())
        data_dim = self.MultiHead_3(data_dim)
        data_dim = F.relu(data_dim)
        print(data_dim.size())
        data_dim = data_dim.view(batch_size,-1)
        print(data_dim.size())
        # fc and dropout.
        data_dim = F.dropout(data_dim)
        output = self.fc_final(data_dim)
        output = torch.sigmoid(output)
        return output


class MultiHead(NN.Module):
    def __init__(self, model_dim=256, output_dim=128, num_head=8, dropout=0.5):
        super(MultiHead, self).__init__()
        self.dim_per_head = model_dim // num_head
        self.num_head = num_head
        self.linear_q = NN.Linear(model_dim, self.dim_per_head * num_head)
        self.linear_k = NN.Linear(model_dim, self.dim_per_head * num_head)
        self.linear_v = NN.Linear(model_dim, self.dim_per_head * num_head)

        self.product_attention = ScaledDotProductAttention(dropout)
        self.fc = NN.Linear(model_dim, output_dim)
        self.dropout = dropout

        self.layer_norm = NN.LayerNorm(model_dim,elementwise_affine=False)
    def forward(self, x):
        residual = x
        batch_size = x.size(0)
        # linear projection
        key = self.linear_k(x)
        value = self.linear_v(x)
        query = self.linear_q(x)

        # reshape
        key = key.view(batch_size * self.num_head, -1, self.dim_per_head)
        value = value.view(batch_size * self.num_head, -1, self.dim_per_head)
        query = query.view(batch_size * self.num_head, -1, self.dim_per_head)

        # attention
        context = self.product_attention(query, key, value, 8)
        # concat
        context = context.view(residual.size())
        # residual
        context += residual
        # layer normal
        context = self.layer_norm(context)
        # fc
        context = self.fc(context)

        return context


class ScaledDotProductAttention(NN.Module):
    def __init__(self, dropout=0.5):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = dropout

    def forward(self, q, k, v, scale=None):
        attention = torch.bmm(q, k.transpose(1, 2))  # Q*K
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=2)  # softmax
        attention = F.dropout(attention)  # dropout
        context = torch.bmm(attention, v)  # attention
        return context

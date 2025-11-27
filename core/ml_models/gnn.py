# """ Pytorch implemenation of bipartite GNN."""

##Changes
""" Pytorch implemenation of GNN."""
##End changes

import torch
import torch.nn.functional as F



# class GraphLayerAtt(torch.nn.Module):
#     """Graph convolutional layer with attention.

#     Parameters
#     ----------
#     dims_in: tuple
#         3-element tuple containing input dimension for supply nodes, demand nodes, and edges,
#         respectively.
#     dims_out: tuple
#         3-element tuple containing output dimension for supply nodes, demand nodes, and edges,
#         respectively.
#     weight_init_func: function, optional
#         Weight initialization function.

#     """

#     def __init__(self, dims_in, dims_out, weight_init_func=None):
#         super(GraphLayerAtt, self).__init__()

#         in_s, in_d, in_e = dims_in
#         out_s, out_d, out_e = dims_out

#         self.dense_ss = torch.nn.Linear(in_s, out_s)
#         self.dense_se = torch.nn.Linear(in_e, out_s, bias=False)
#         self.attention_se = torch.nn.Linear(in_e, 1)

#         self.dense_dd = torch.nn.Linear(in_d, out_d)
#         self.dense_de = torch.nn.Linear(in_e, out_d, bias=False)
#         self.attention_de = torch.nn.Linear(in_e, 1)

#         self.dense_ee = torch.nn.Linear(in_e, out_e)
#         self.dense_es = torch.nn.Linear(in_s, out_e, bias=False)
#         self.dense_ed = torch.nn.Linear(in_d, out_e, bias=False)

#         if weight_init_func is not None:
#             weight_init_func(self.dense_ss.weight)
#             weight_init_func(self.dense_se.weight)
#             weight_init_func(self.dense_dd.weight)
#             weight_init_func(self.dense_de.weight)
#             weight_init_func(self.dense_ee.weight)
#             weight_init_func(self.dense_es.weight)
#             weight_init_func(self.dense_ed.weight)

#     def forward(self, x_s, x_d, x_e):
#         # x_s: b x n x 1, x_d: b x m x 1, x_e: b x n x m x 2
#         h_s = self.dense_ss(x_s) + torch.sum(
#             self.dense_se(x_e) * self.attention_se(x_e).softmax(dim=2), dim=2
#         )

#         h_d = self.dense_dd(x_d) + torch.sum(
#             self.dense_de(x_e) * self.attention_de(x_e).softmax(dim=1), dim=1
#         )
#         h_e = (
#             self.dense_ee(x_e)
#             + self.dense_es(x_s)[:, :, None, :]
#             + self.dense_ed(x_d)[:, None, :, :]
#         )
#         return h_s, h_d, h_e


# class GraphNNAtt(torch.nn.Module):
#     """GNN with attention.

#     Parameters
#     ----------
#     dims_in: tuple
#         3-element tuple containing input dimension for supply nodes, demand nodes, and edges,
#         respectively.
#     conv_dims: tuple
#         Output dimensions of graph convolutional layers. The length of the tuple defines the
#         number of convolutional layers.
#     dense_dims: tuple
#         Output dimensions of dense layers. The length of the tuple defines the number of dense
#         layers.
#     dim_out: int
#         Edge output dimension.
#     weight_init_func: function, optional
#         Weight initialization function.

#     """

#     def __init__(
#         self,
#         dims_in,
#         conv_dims,
#         dense_dims,
#         dim_out,
#         weight_init_func=None,
#     ):
#         super(GraphNNAtt, self).__init__()

#         # convolution layers
#         self.conv = torch.nn.ModuleList()
#         self.conv.append(
#             GraphLayerAtt(dims_in, conv_dims[0], weight_init_func=weight_init_func)
#         )
#         for i in range(1, len(conv_dims)):
#             self.conv.append(
#                 GraphLayerAtt(
#                     conv_dims[i - 1], conv_dims[i], weight_init_func=weight_init_func
#                 )
#             )

#         # dense layers
#         self.dense = torch.nn.ModuleList()
#         if len(dense_dims) >= 1:
#             self.dense.append(torch.nn.Linear(conv_dims[-1][-1], dense_dims[0]))
#             for i in range(1, len(dense_dims)):
#                 self.dense.append(torch.nn.Linear(dense_dims[i - 1], dense_dims[i]))

#         # output layer
#         self.output_dim = dim_out
#         if len(dense_dims) >= 1:
#             self.out = torch.nn.Linear(dense_dims[-1], dim_out)
#         else:
#             self.out = torch.nn.Linear(conv_dims[-1][-1], dim_out)

#     def forward(self, x_s, x_d, x_e):
#         for conv_layer in self.conv:
#             x_s, x_d, x_e = conv_layer(x_s, x_d, x_e)
#             x_s, x_d, x_e = F.relu(x_s), F.relu(x_d), F.relu(x_e)

#         for dense_layer in self.dense:
#             x_e = F.relu(dense_layer(x_e))

#         output = self.out(x_e)

#         return output



##Changes


class GraphLayerAtt(torch.nn.Module):
    """Graph convolutional layer with attention.

    Parameters
    ----------
    dims_in: tuple
        2-element tuple containing input dimension for nodes, and arcs,
        respectively.
    dims_out: tuple
        2-element tuple containing output dimension for nodes, and arcs,
        respectively.
    weight_init_func: function, optional
        Weight initialization function.

    """

    def __init__(self, dims_in, dims_out, weight_init_func=None):
        super(GraphLayerAtt, self).__init__()

        in_v, in_a = dims_in
        out_v, out_a = dims_out

        self.dense_vv = torch.nn.Linear(in_v, out_v)
        self.dense_va = torch.nn.Linear(in_a, out_v, bias=False)
        self.attention_va = torch.nn.Linear(in_a, 1)

        self.dense_aa = torch.nn.Linear(in_a, out_a)
        self.dense_av = torch.nn.Linear(in_v, out_a, bias=False)

        if weight_init_func is not None:
            weight_init_func(self.dense_vv.weight)
            weight_init_func(self.dense_va.weight)
            weight_init_func(self.dense_aa.weight)
            weight_init_func(self.dense_av.weight)


    def forward(self, x_v, x_a, arc_index):
        # x_v:  (n+1) x 1, x_a:  m x 2

        src, dst = arc_index
        h_self = self.dense_vv(x_v)

        e = self.attention_va(x_a).squeeze(-1)   # (m,)
        # e is attention logits per edge

        # compute softmax per destination node
        # Step 1: max per destination
        num_nodes = x_v.size(0)
        max_per_dst = x_v.new_full((num_nodes,), float('-inf'))
        max_per_dst = max_per_dst.index_reduce(0, dst, e, reduce='amax')
        # Step 2: subtract and exponentiate
        exp_e = torch.exp(e - max_per_dst[dst])
        # Step 3: normalize
        denom = torch.zeros_like(max_per_dst).index_add(0, dst, exp_e)
        attn = exp_e / denom[dst]
        attn = attn.unsqueeze(-1)
        msg_from_arcs = self.dense_va(x_a) * attn


        # Aggregate both into destination nodes
        h_msg = torch.zeros_like(h_self)
        h_msg = h_msg.index_add(0, dst, msg_from_arcs)

        h_v = h_msg + h_self
        # edge update
        h_a = (
            self.dense_aa(x_a)
            + self.dense_av(x_v[src])
            + self.dense_av(x_v[dst])
        )

        return h_v, h_a


class GraphNNAtt(torch.nn.Module):
    """GNN with attention.

    Parameters
    ----------
    dims_in: tuple
        2-element tuple containing input dimension for nodes, and arcs,
        respectively.
    conv_dims: tuple
        Output dimensions of graph convolutional layers. The length of the tuple defines the
        number of convolutional layers.
    dense_dims: tuple
        Output dimensions of dense layers. The length of the tuple defines the number of dense
        layers.
    dim_out: int
        Arc output dimension.
    weight_init_func: function, optional
        Weight initialization function.

    """

    def __init__(
        self,
        dims_in,
        conv_dims,
        dense_dims,
        dim_out,
        weight_init_func=None,
    ):
        super(GraphNNAtt, self).__init__()

        # convolution layers
        self.conv = torch.nn.ModuleList()
        self.conv.append(
            GraphLayerAtt(dims_in, conv_dims[0], weight_init_func=weight_init_func)
        )
        for i in range(1, len(conv_dims)):
            self.conv.append(
                GraphLayerAtt(
                    conv_dims[i - 1], conv_dims[i], weight_init_func=weight_init_func
                )
            )

        # dense layers
        self.dense = torch.nn.ModuleList()
        if len(dense_dims) >= 1:
            self.dense.append(torch.nn.Linear(conv_dims[-1][-1], dense_dims[0]))
            for i in range(1, len(dense_dims)):
                self.dense.append(torch.nn.Linear(dense_dims[i - 1], dense_dims[i]))

        # output layer
        self.output_dim = dim_out
        if len(dense_dims) >= 1:
            self.out = torch.nn.Linear(dense_dims[-1], dim_out)
        else:
            self.out = torch.nn.Linear(conv_dims[-1][-1], dim_out)

    def forward(self, x_v, x_a, arc_index):
        for conv_layer in self.conv:
            x_v, x_a = conv_layer(x_v, x_a, arc_index)
            x_v, x_a = F.relu(x_v), F.relu(x_a)

        for dense_layer in self.dense:
            x_a = F.relu(dense_layer(x_a))

        output = self.out(x_a)

        return output
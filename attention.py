import torch.nn as nn
import torch


class BasicAttention(nn.Module):
    def __init__(self,
                 heads,
                 size_per_head,
                 key_size=None,
                 mask_right=False,
                 score_func='scaled_dot',
                 drop_rate=0.,
                 bias=True
                 ):

        super(BasicAttention, self).__init__()

        self.num_heads = heads
        self.size_per_head = size_per_head

        self.out_dim = size_per_head*heads
        self.key_size = key_size if key_size else size_per_head
        self.mask_right = mask_right

        self.q_w = nn.Linear(self.key_size * self.num_heads, self.key_size * self.num_heads,bias=bias)

        self.k_w = nn.Linear(self.key_size * self.num_heads, self.key_size * self.num_heads,bias=bias)

        self.v_w = nn.Linear(self.key_size * self.num_heads, self.out_dim,bias=bias)

        self.score_func = score_func
        self.drop_rate = drop_rate

    def forward(self, input, mask=None):
        '''
        batch-first is needed
        :param q_embd: [?,q_len,q_embd_size] or [?,q_embd_size]
        :param k_embd: [?,k_len,k_embd_size] or [?,k_embd_size]
        :param v_embd: [?,v_len,v_embd_size] or [?,v_embd_size]
        :return: [?,q_len,output_hidden_size*num_heads]
        '''
        q_embd = input[:1]
        k_embd = input[:1]
        v_embd = input[:1]
        if len(q_embd.shape) == 2:
            q_embd = torch.unsqueeze(q_embd, 1)
        if len(k_embd.shape) == 2:
            k_embd = torch.unsqueeze(k_embd, 1)
        if len(v_embd.shape) == 2:
            v_embd = torch.unsqueeze(v_embd, 1)
        batch_size = q_embd.shape[0]
        q_len = q_embd.shape[1]
        k_len = k_embd.shape[1]
        v_len = v_embd.shape[1]
        #     make sure k_len==v_len
        assert k_len == v_len

        # get q,k,v
#         if self.is_q:
#             q = self.q_w(q_embd).view(batch_size, q_len, self.num_heads, self.q_embd_size // self.num_heads)
#             q = q.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.q_embd_size // self.num_heads)
#         else:
        q = self.q_w(q_embd).view(batch_size, q_len, self.num_heads, self.key_size)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.key_size)
        k = self.k_w(k_embd).view(batch_size, k_len, self.num_heads, self.key_size)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.key_size)
        v = self.v_w(v_embd).view(batch_size, v_len, self.num_heads, self.size_per_head )
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, v_len, self.size_per_head)

        # get score
        if isinstance(self.score_func, str):
            if self.score_func == "dot":
                score = torch.bmm(q, k.permute(0, 2, 1))

            elif self.score_func == "scaled_dot":
                temp = torch.bmm(q, k.permute(0, 2, 1))
                score = torch.div(temp, math.sqrt(self.key_size))

            else:
                raise RuntimeError('invalid score function')
        elif callable(self.score_func):
            try:
                score = self.score_func(q, k)
            except Exception as e:
                print("Exception :", e)
        if mask is not None:
            mask = mask.bool().unsqueeze(1)
            score = score.masked_fill(~mask, -np.inf)
        score = nn.functional.softmax(score, dim=-1)
        score = nn.functional.dropout(score, p=self.drop_rate, training=self.training)

        # get output
        output = torch.bmm(score, v)
        heads = torch.split(output, batch_size)
        output = torch.cat(heads, -1)

        return output
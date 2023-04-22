from torch import nn


class TokenEmbeddings(nn.Embedding):
    def __init__(self,vocab_size , d_model):
        super(TokenEmbeddings,self).__init__(vocab_size, d_model ,padding_idx=1)

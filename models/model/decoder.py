from torch import nn
from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    def __init__(self,d_model,max_len,dec_vocab_size, n_head ,drop_prob,n_layers ,ffn_hidden,device):
        super(Decoder,self).__init__()
        self.emb = TransformerEmbedding(max_len=max_len,
                                        vocab_size= dec_vocab_size,
                                        d_model=d_model,
                                        drop_prob=drop_prob,
                                        device = device)
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model ,dec_vocab_size)

    def forward(self,trg , enc_src , trg_mask , src_mask):
        trg = self.emb(trg)
        for layer in self.layers:
            trg =layer(trg,enc_src ,trg_mask,src_mask)
        output = self.linear(trg)
        return output    




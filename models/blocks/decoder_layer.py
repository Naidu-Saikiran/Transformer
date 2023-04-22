from torch import nn
from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model ,ffn_hidden , n_head ,drop_prob):
        super(DecoderLayer,self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model ,n_head=n_head)
        self.norm1 = LayerNorm(d_model = d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        
        self.enc_dec_attention = MultiHeadAttention(d_model,n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model ,hidden = ffn_hidden,drop_prob = drop_prob)
        self.norm3 = LayerNorm(d_model =d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)
    def forward(self,dec ,enc, t_mask ,s_mask):
        _x =dec
        x = self.attention(q=dec ,k= dec ,v=dec , mask=t_mask)
        x = self.norm1(x)
        x = self.dropout1(x+_x)
        _x = x
        x = self.enc_dec_attention(q=enc ,k=enc ,v=enc ,mask =s_mask)
        x = self.dropout2(x)
        x = self.norm2(x+_x)

        _x= x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x+_x)
        return x

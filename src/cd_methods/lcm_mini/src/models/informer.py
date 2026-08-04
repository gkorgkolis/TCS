import math

import torch
import torch.nn as nn
import torch.nn.functional as F

#  https://github.com/martinwhl/Informer-PyTorch-Lightning

class Informer(nn.Module):
    def __init__(
        self,
        d_model=32, # model dimension
        n_heads=4, # number of attention heads
        num_encoder_layers=1,
        d_ff=64, # feed-forward layer size
        dropout=0.05, # dropout rate
        activation="gelu", # GELU activation function as in the informer paper
        output_attention=False, # whether to return the attention weights
        distil=True, # distiled attention
        max_length=500, # maximum length of the input time series 
        n_vars=12,
        max_lags=3,
        corr_input=False, # by default, no correlation injection
        regression_head=False, # by default, no regression head
        **kwargs
    ):
        super(Informer, self).__init__()

        self.corr_input = corr_input
        self.regression_head = regression_head
        self.n_vars = n_vars
        self.max_lags = max_lags

        self.norm = nn.BatchNorm1d(d_ff)
        self.activation = nn.ELU()

        self.enc_embedding = DataEmbedding( # use c_in = n_vars input channels 
            n_vars, d_model, dropout, max_length=max_length, kernel_size=3
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        #d_model,
                        #n_heads,
                        #output_attention=output_attention,
                        FullAttention(
                            None,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                        mix=False,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_encoder_layers)
            ],
            (
                [SelfAttentionDistil(d_model) for _ in range(num_encoder_layers - 1)]
                if distil
                else None
            ),
            nn.LayerNorm(d_model),
        )


        if corr_input:
           in_dim = d_model + (n_vars**2 * max_lags) # append the (flattened) correlation matrix or te_input (dim (n_vars, n_vars, max_lags)) to the input
        else:
           in_dim = d_model  # only use the last hidden state
       
        self.fc1 = torch.nn.Linear(in_dim, d_ff) # dim is (in_dim, d_ff)
        self.fc2 = torch.nn.Linear(d_ff, n_vars**2 * max_lags) # (d_ff, n_vars**2 * max_lags)
        if regression_head:
            self.fc3 = torch.nn.Linear(d_ff, 1) # dim is just (d_ff, 1)

    # reshaping function 
    def reformat(self, x):
        return torch.reshape(x, (x.shape[0], self.n_vars, self.n_vars, self.max_lags)) # reshape to (n_vars, n_vars, max_lags) tensor

    # forward pass
    def forward(
        self,
        x_enc, # input time series data embedding (sum of token and positional embeddings)
        ):
        # Debug
        #print(f"forward pass, x_enc: {x_enc}")
        # Unpacking the input
        if self.corr_input:
            x_enc, corr = x_enc  # Expecting 2 
            #print(f'Unpacked: x_enc shape: {x_enc.shape}, corr shape: {corr.shape}')
        else:
            corr = None

        ## Embedding and encoding
        enc_out = self.enc_embedding(x_enc)
        enc_out, attentions = self.encoder(enc_out)

        if self.corr_input:
            corr = corr.view(corr.shape[0], -1)  # flatten
        
        # Concatenate relevant inputs
        if self.corr_input:
            inp = torch.cat((enc_out[:, -1, :], corr), dim=1)
        else:
            inp = enc_out[:, -1, :]
        
        # Debugging
        # shapes after concat
        #print("Shapes after concat: \n")
        #print(f'corr.shape: {corr.shape}')
        #print(f'te.shape: {te.shape}')
        #print(f'prior.shape: {prior.shape}')
        #print(f'enc_out.shape: {enc_out.shape}')
        #print(f'inp.shape: {inp.shape}')

        hidden1 = self.activation(self.norm(self.fc1(inp)))
        hidden2 = self.fc2(hidden1)

        if self.regression_head:
            reg_out = self.fc3(hidden1)
            #####
            edge_labels = torch.sigmoid(reg_out)
            edge_labels = (edge_labels > 0.05).float()

        if self.regression_head:
            return (self.reformat(hidden2), reg_out)
        else:
            return self.reformat(hidden2)
        

class Encoder(nn.Module):
    def __init__(self, attention_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attention_layers = nn.ModuleList(attention_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(self, x):
        attentions = []
        if self.conv_layers is not None:
            for attention_layer, conv_layer in zip(
                self.attention_layers, self.conv_layers
            ):
                x, attention = attention_layer(x)
                x = conv_layer(x)
                attentions.append(attention)
            x, attention = self.attention_layers[-1](x)
            attentions.append(attention)
        else:
            for attention_layer in self.attention_layers:
                x, attention = attention_layer(x)
                attentions.append(attention)

        if self.norm is not None:
            x = self.norm(x)
        return x, attentions


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x):
        new_x, attention = self.attention(x, x, x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attention


class AttentionLayer(nn.Module):
    # performs the QKV mapping and then runs the full attention operation
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention # here the attention layer is passed from FullAttention
        self.query_attention = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads
        self.mix = mix 

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_attention(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attention = self.inner_attention(queries, keys, values)

        out = out.view(B, L, -1) # reshape attention output

        return self.out_projection(out), attention


class FullAttention(nn.Module):
    # Vanilla full attention (self-attention) computation
    def __init__(
        self, scale=None, attention_dropout=0.1, output_attention=False, **kwargs
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        # Vaswani et al. scaling factor
        scale = self.scale or 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) # dot product of queries and keys

        A = self.dropout(torch.softmax(scale * scores, dim=-1)) # output weights, softmax to the scaled scores
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        return V.contiguous(), None


class SparseAttention(nn.Module):
    def __init__(self):
        super(SparseAttention, self).__init__()

    def forward(self, x):
        raise NotImplementedError('Sparse Attention is not implemented yet in the SpaceTimeFormer Module.')


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, max_length=5000, kernel_size=3):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model, kernel_size=kernel_size)
        self.position_embedding = PositionalEmbedding(d_model, max_length=max_length)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        a = self.value_embedding(x)
        b = self.position_embedding(x)
        x = a + b  
        return self.dropout(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_length=5000):
        super(PositionalEmbedding, self).__init__()
        embedding = torch.zeros(max_length, d_model) # shape is [max_length, d_model]

        position = torch.arange(0, max_length).float().unsqueeze(1) # shape is [max_length, 1]
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        embedding[:, 0::2] = torch.sin(position * div_term) # even terms
        embedding[:, 1::2] = torch.cos(position * div_term) # odd terms

        embedding = embedding.unsqueeze(0) # [1, max_length, d_model]
        self.register_buffer("embedding", embedding)

    def forward(self, x):
        return self.embedding[:, : x.size(1)] # forward pass is of shape [1, x.size(1), d_model], just returns the positional encoding

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, kernel_size=3):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.token_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="circular",
        )
        # perform Kaiming-He weight initialization for the conv weights
        nn.init.kaiming_normal_(
            self.token_conv.weight, mode="fan_in", nonlinearity="leaky_relu"
        )

    def forward(self, x):
        # somehow sometimes the dimension breaks. Fix that here:
        l = x.shape[1]
        out = self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)
        if out.shape[1] != l:
            out = out[:, :l, :] # output shape is [batch_size, seq_len, d_model]
        return out


class SelfAttentionDistil(nn.Module):
    def __init__(self, c_in):
        super(SelfAttentionDistil, self).__init__()
        self.conv = nn.Conv1d(
            c_in, c_in, kernel_size=3, padding=2, padding_mode="circular"
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        x = torch.transpose(x, 1, 2)
        return x
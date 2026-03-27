import torch
import torch.nn as nn
import torch.optim as optim
import math

# CODIFICACIÓN POSICIONAL

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2]=torch.sin(position * div_term)
        pe[:, 1::2]=torch.cos(position * div_term)
        pe=pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


# TRANSFORMER MODEL

class TransformerModel(nn.Module):
    def __init__(self, ntoken, d_model, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type='Transformer'
        self.pos_encoder=PositionalEncoding(d_model)
        encoder_layers=nn.TransformerEncoderLayer(d_model, nhead, nhid, dropout)
        self.transformer_encoder=nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder=nn.Embedding(ntoken, d_model)
        self.d_model=d_model
        self.decoder=nn.Linear(d_model, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange=0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src=self.encoder(src) * math.sqrt(self.d_model)
        src=self.pos_encoder(src)
        output=self.transformer_encoder(src, src_mask)
        output=self.decoder(output)
        return output


#EJECUCIÓN

if __name__=="__main__":
    # Parámetros del modelo
    ntoken = 20    # vocabulario de 20 tokens
    d_model = 32   # dimensión del embedding
    nhead = 4      # número de cabezas de atención
    nhid = 64      # tamaño del feed-forward
    nlayers = 2    # capas del encoder
    
    model=TransformerModel(ntoken, d_model, nhead, nhid, nlayers)

    # Crear una secuencia de 10 tokens aleatorios
    src = torch.randint(0, ntoken, (10, 1))

    # Crear máscara (aquí no bloquea nada)
    src_mask = torch.zeros((10, 10))

    # Ejecutar el modelo
    output = model(src, src_mask)

    # Mostrar resultados
    print("Secuencia de entrada:\n", src)
    print("\nSalida del modelo (predicciones):\n", output)
    print("\nTamaño de salida:", output.shape)
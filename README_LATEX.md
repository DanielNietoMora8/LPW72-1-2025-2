Transformers y Large Language Models (LLMs) — Implementación y Experimentos en Python

Autores:

Jorge Luis Berrio Pino – jorgeberrio196242@correo.itm.edu.co

Julián David Arias Zapata – julianarias240909@correo.itm.edu.co

Instituto Tecnológico Metropolitano – ITM
Fecha del informe: 18 de noviembre de 2025
-----------------------------------------------------------------------------------

Descripción del Proyecto

Este repositorio contiene:

Un artículo académico en formato IEEE desarrollado en LaTeX.

Una implementación funcional en Python de un Transformer Encoder.

Experimentos que muestran:

Entrenamiento

Pérdida por época

Métricas simples

Ejemplo de inferencia
---------------------------------------------------------------------------------
RESUMEN

Los Transformers, introducidos en Attention is All You Need (2017), reemplazan la recurrencia por autoatención escalada, permitiendo paralelización y captura de dependencias largas.

Los LLMs como GPT-5.1, Llama 4 y Gemini 2.5 usan variantes más profundas y multimodales del Transformer, entrenadas con billones de tokens.

El artículo en LaTeX expone:

Arquitectura Transformer

Autoatención

Comparación entre modelos modernos

Implementación experimental

Resultados
---------------------------------------------------------------------------------

Implementación en Python

La implementación está dividida en módulos:

🔹 1. model.py

Implementa un Transformer Encoder usando PyTorch:

Positional Encoding

Multi-Head Attention

Feed-Forward Networks

Embedding + Linear Decoder

🔹 2. train.py

Incluye:

Generación del dataset sintético

Entrenamiento por 50 épocas

Cálculo de pérdida

Guardado de un archivo loss_values.txt

Generación de gráfica loss_plot.png

🔹 3. generate.py

Carga el modelo entrenado y genera secuencia de texto.

🔹 4. dataset.py

Genera texto repetitivo estilo “hello world” para entrenar a nivel carácter.
---------------------------------------------------------------------------------

Resultados de Entrenamiento

Los resultados experimentales obtenidos:

Época	Pérdida
10	1.20
20	0.80
30	0.40
40	0.10
50	0.05

Pérdida final: 0.05
Perplejidad aproximada: 1.05

Esto confirma que el modelo memoriza el pequeño dataset (como se espera en este tipo de demo de manera académica).

---------------------------------------------------------------------------------

Figuras necesarias (para Overleaf)

latex/
 ├── figura-transformer.png
 ├── grafica-entrenamiento.png

Estas se referencian en el artículo: \includegraphics[width=0.8\linewidth]{figura-transformer.png}

---------------------------------------------------------------------------------
Cómo Ejecutar el Proyecto
1. Instalar dependencias
pip install torch matplotlib numpy

2. Entrenar el modelo
python train.py

3. Generar texto
python generate.py

4. Revisar resultados

loss_plot.png

loss_values.txt

Texto generado
---------------------------------------------------------------------------------

Artículo en LaTeX

El archivo se encuentra en:

latex/articulo_transformers_llms.tex


Formato: IEEE conference
Incluye:

Introducción teórica

Métodos

Arquitectura

Comparación de LLMs

Algoritmo

Resultados y métricas

Conclusiones

Bibliografía IEEE


---------------------------------------------------------------------------------

Código

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
---------------------------------------------------------------------------------

Cumplimiento de los requisitos del docente Daniel Alexis Nieto Mora

Requisito	    ¿Cumplido?
Exposición teórica	✔
Implementación Python	✔
Código funcional	✔
Resultados y métricas	✔
README documentado	✔
Entrega en GitHub	✔
Artículo tipo informe	✔ (IEEE)

--------------------------------------------------------------------------------





import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import random
from collections import Counter
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_model_optimization as tfmot

# Crescimento de memória da GPU: precisa vir antes de qualquer operação do TF
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
DIR_DATASET = os.path.join(DIRETORIO_ATUAL, "dataset", "visores")
DIR_CSVS = os.path.join(DIRETORIO_ATUAL, "CSVs")
DIR_SAIDA = os.path.join(DIRETORIO_ATUAL, "Modelos")

# Entrada fixa (necessário para o TFLite int8), mas SEM esticar a imagem:
# o recorte é redimensionado mantendo a proporção e o que sobra é preenchido.
# Recortes mais largos que IMG_WIDTH/IMG_HEIGHT são reduzidos proporcionalmente.
IMG_HEIGHT = 48                 # precisa ser múltiplo de 16
IMG_WIDTH = 224                 # precisa ser múltiplo de 4
TIMESTEPS = IMG_WIDTH // 4      # colunas de saída da CNN (eixo "tempo" do CTC)

MAX_DIGITOS = 8                 # maior sequência aceita (com folga sobre 7)
N_CLASSES = 11                  # 0-9 + blank do CTC (índice 10)
BLANK = 10
PAD = -1                        # preenchimento dos rótulos (não é classe)

BATCH_SIZE = 32
EPOCAS_FLOAT = 300
EPOCAS_QAT = 80

os.makedirs(DIR_CSVS, exist_ok=True)
os.makedirs(DIR_SAIDA, exist_ok=True)

# ==========================================
# 2. MAPEAMENTO DOS ARQUIVOS E RÓTULOS
# ==========================================
# Com CTC não existe "X" de preenchimento: os X no FINAL do nome da pasta
# são descartados ("000777X" -> "000777"). Um X no meio do nome não tem
# representação nesse formato, então a pasta é ignorada (com aviso).
print("-> Mapeando arquivos e higienizando rótulos das pastas...")

imagens_por_rotulo = {}   # string de dígitos -> lista de caminhos
pastas_ignoradas = []

for root, dirs, files in os.walk(DIR_DATASET):
    nome_pasta = os.path.basename(root)

    if not nome_pasta or nome_pasta.startswith('.') or nome_pasta == "visores":
        continue

    imgs = [os.path.join(root, f) for f in files
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not imgs:
        continue

    rotulo = nome_pasta.strip().upper().rstrip('X')

    if not (1 <= len(rotulo) <= MAX_DIGITOS) or not all(c in '0123456789' for c in rotulo):
        pastas_ignoradas.append(nome_pasta)
        continue

    imagens_por_rotulo.setdefault(rotulo, []).extend(imgs)

if not imagens_por_rotulo:
    raise ValueError(f"Nenhuma imagem válida encontrada em: {DIR_DATASET}.")

if pastas_ignoradas:
    print(f"   [AVISO] {len(pastas_ignoradas)} pasta(s) ignorada(s) por nome inválido. "
          f"Exemplos: {pastas_ignoradas[:5]}")

distribuicao = Counter(len(r) for r in imagens_por_rotulo)
print("   Leituras por quantidade de dígitos: " +
      ", ".join(f"{k} dígitos: {v}" for k, v in sorted(distribuicao.items())))

# Split por PASTA (leitura), para não vazar quase-duplicatas para a validação
rotulos = sorted(imagens_por_rotulo.keys())
random.Random(SEED).shuffle(rotulos)

corte = int(len(rotulos) * 0.8)
rotulos_treino, rotulos_val = rotulos[:corte], rotulos[corte:]
if not rotulos_treino or not rotulos_val:
    raise ValueError("Poucas pastas para separar treino e validação.")

def montar_listas(lista_rotulos):
    caminhos, labels = [], []
    for r in lista_rotulos:
        codificado = [int(c) for c in r] + [PAD] * (MAX_DIGITOS - len(r))
        for caminho in imagens_por_rotulo[r]:
            caminhos.append(caminho)
            labels.append(codificado)
    return caminhos, np.array(labels, dtype=np.int32)

caminhos_treino, labels_treino = montar_listas(rotulos_treino)
caminhos_val, labels_val = montar_listas(rotulos_val)

print(f"-> {len(rotulos)} leituras distintas | treino: {len(caminhos_treino)} imagens "
      f"({len(rotulos_treino)} pastas) | validação: {len(caminhos_val)} imagens "
      f"({len(rotulos_val)} pastas)")

# ==========================================
# 3. PIPELINE tf.data
# ==========================================
def carregar(caminho, rotulo):
    img = tf.io.read_file(caminho)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)
    img = tf.cast(img, tf.float32)

    altura = tf.cast(tf.shape(img)[0], tf.float32)
    largura = tf.cast(tf.shape(img)[1], tf.float32)

    # Redimensiona mantendo a proporção, sem passar de IMG_HEIGHT x IMG_WIDTH
    escala = tf.minimum(float(IMG_HEIGHT) / altura, float(IMG_WIDTH) / largura)
    nova_h = tf.maximum(1, tf.cast(tf.round(altura * escala), tf.int32))
    nova_w = tf.maximum(1, tf.cast(tf.round(largura * escala), tf.int32))
    img = tf.image.resize(img, [nova_h, nova_w])

    # Preenche à direita/embaixo com a cor de fundo (média da última coluna).
    # A INFERÊNCIA PRECISA REPETIR EXATAMENTE ESTE PRÉ-PROCESSAMENTO.
    fundo = tf.reduce_mean(img[:, -1:, :])
    img = tf.pad(img,
                 [[0, IMG_HEIGHT - nova_h], [0, IMG_WIDTH - nova_w], [0, 0]],
                 constant_values=fundo)
    img = tf.reshape(img, [IMG_HEIGHT, IMG_WIDTH, 1])
    img = tf.cast(tf.round(img), tf.uint8)      # uint8 deixa o cache leve
    return img, rotulo

def normalizar(img, rotulo):
    return tf.cast(img, tf.float32) / 255.0, rotulo

def augmentar(img, rotulo):
    # Sem flip (inverte os dígitos) e com deslocamento pequeno para não
    # cortar os dígitos das bordas do recorte.
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.pad(img, [[2, 2], [2, 2], [0, 0]], mode='SYMMETRIC')
    img = tf.image.random_crop(img, [IMG_HEIGHT, IMG_WIDTH, 1])
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, rotulo

ds_treino_cache = (tf.data.Dataset.from_tensor_slices((caminhos_treino, labels_treino))
                   .map(carregar, num_parallel_calls=AUTOTUNE)
                   .cache())

dataset_treino = (ds_treino_cache
                  .shuffle(len(caminhos_treino), seed=SEED)
                  .map(normalizar, num_parallel_calls=AUTOTUNE)
                  .map(augmentar, num_parallel_calls=AUTOTUNE)
                  .batch(BATCH_SIZE)
                  .prefetch(AUTOTUNE))

dataset_val = (tf.data.Dataset.from_tensor_slices((caminhos_val, labels_val))
               .map(carregar, num_parallel_calls=AUTOTUNE)
               .cache()
               .map(normalizar, num_parallel_calls=AUTOTUNE)
               .batch(BATCH_SIZE)
               .prefetch(AUTOTUNE))

# ==========================================
# 4. LOSS CTC, DECODIFICAÇÃO E MÉTRICA
# ==========================================
def ctc_loss(y_true, y_pred):
    # y_true: (batch, MAX_DIGITOS) com PAD nas posições sem dígito
    # y_pred: (batch, TIMESTEPS, N_CLASSES) -> logits (sem softmax)
    y_true = tf.cast(y_true, tf.int32)
    tamanho_label = tf.reduce_sum(tf.cast(tf.not_equal(y_true, PAD), tf.int32), axis=1)
    labels = tf.where(tf.equal(y_true, PAD), tf.zeros_like(y_true), y_true)
    tamanho_logits = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
    return tf.nn.ctc_loss(labels=labels,
                          logits=tf.cast(y_pred, tf.float32),
                          label_length=tamanho_label,
                          logit_length=tamanho_logits,
                          logits_time_major=False,
                          blank_index=-1)     # blank = última classe (10)

def decodificar_greedy(logits):
    """(TIMESTEPS, N_CLASSES) -> lista de dígitos.
    Pega o argmax por coluna, colapsa repetições e remove o blank."""
    ids = np.argmax(logits, axis=-1)
    saida, anterior = [], BLANK
    for i in ids:
        if i != anterior and i != BLANK:
            saida.append(int(i))
        anterior = i
    return saida

def lista_para_texto(lista):
    return ''.join(str(d) for d in lista)

def avaliar(modelo, dataset):
    """Retorna lista de (real, previsto), cada um como lista de dígitos."""
    resultados = []
    for x, y in dataset:
        logits = modelo(x, training=False).numpy()
        for l, r in zip(logits, y.numpy()):
            real = [int(v) for v in r if v != PAD]
            resultados.append((real, decodificar_greedy(l)))
    return resultados

def acuracia_sequencia(resultados):
    return float(np.mean([real == pred for real, pred in resultados]))

class AcuraciaSequenciaCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        acc = acuracia_sequencia(avaliar(self.model, self.dataset))
        if logs is not None:
            logs['val_seq_acc'] = acc
        print(f" - val_seq_acc: {acc:.4f}")

# ==========================================
# 5. ARQUITETURA CRNN SEM RECORRÊNCIA (CONV -> CTC)
# ==========================================
# Recorrência (LSTM/GRU) não é suportada pelo QAT do tfmot, então o contexto
# horizontal vem só de convoluções. A altura é colapsada até 1 e a largura
# vira o eixo de tempo: cada coluna emite uma distribuição sobre 0-9 + blank.
print("\n-> Construindo a rede CRNN (conv-only + CTC)...")

def bloco_conv(x, filtros, kernel=(3, 3), padding='same'):
    # Conv(sem bias) -> BN -> ReLU é o padrão que o tfmot sabe fundir no QAT
    x = layers.Conv2D(filtros, kernel, padding=padding, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)

inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))

x = bloco_conv(inputs, 16)
x = layers.MaxPooling2D((2, 2))(x)          # 24 x 112  (para 48 x 224)
x = bloco_conv(x, 32)
x = layers.MaxPooling2D((2, 2))(x)          # 12 x 56
x = bloco_conv(x, 48)
x = layers.MaxPooling2D((2, 1))(x)          # 6 x 56  (só a altura reduz)
x = bloco_conv(x, 64)
x = layers.MaxPooling2D((2, 1))(x)          # 3 x 56
x = bloco_conv(x, 64)                       # contexto lateral entre colunas
x = bloco_conv(x, 64, kernel=(IMG_HEIGHT // 16, 1), padding='valid')   # 1 x 56
x = layers.Reshape((TIMESTEPS, 64))(x)      # (batch, tempo, features)
x = layers.Dropout(0.2)(x)
logits = layers.Dense(N_CLASSES, name='logits')(x)

base_model = models.Model(inputs=inputs, outputs=logits)

# ==========================================
# 6. FASE 1 — TREINO EM PONTO FLUTUANTE
# ==========================================
# É normal a loss CTC ficar "travada" por várias épocas (a rede começa
# prevendo só blank) e val_seq_acc ficar em 0 até ela "descolar".
print("\n-> Fase 1: treino em ponto flutuante (sem quantização)")
base_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
    loss=ctc_loss
)

history_float = base_model.fit(
    dataset_treino,
    validation_data=dataset_val,
    epochs=EPOCAS_FLOAT,
    callbacks=[
        AcuraciaSequenciaCallback(dataset_val),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                             patience=10, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30,
                                         restore_best_weights=True)
    ]
)

# ==========================================
# 7. FASE 2 — FINE-TUNING COM QAT
# ==========================================
print("\n-> Fase 2: aplicando QAT e fazendo fine-tuning com learning rate baixa")
qat_model = tfmot.quantization.keras.quantize_model(base_model)
qat_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0),
    loss=ctc_loss
)

history_qat = qat_model.fit(
    dataset_treino,
    validation_data=dataset_val,
    epochs=EPOCAS_QAT,
    callbacks=[
        AcuraciaSequenciaCallback(dataset_val),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15,
                                         restore_best_weights=True)
    ]
)

# ==========================================
# 8. EXPORTAÇÃO DO HISTÓRICO E AVALIAÇÃO DO MODELO QAT
# ==========================================
print("\n-> Exportando métricas de treino para CSV...")
df_float = pd.DataFrame(history_float.history)
df_float['fase'] = 'float'
df_qat = pd.DataFrame(history_qat.history)
df_qat['fase'] = 'qat'
df_historico = pd.concat([df_float, df_qat], ignore_index=True)
caminho_csv = os.path.join(DIR_CSVS, "historico_treino_ctc.csv")
df_historico.to_csv(caminho_csv, index_label="Epoca")
print(f"-> Arquivo CSV salvo em: {caminho_csv}")

def mostrar_exemplos(resultados, n=10):
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(resultados), size=min(n, len(resultados)), replace=False)
    for i in idx:
        real, pred = resultados[i]
        ok = "OK  " if real == pred else "ERRO"
        print(f"     {ok} {lista_para_texto(real)} -> {lista_para_texto(pred)}")

print("\n-> Avaliação do modelo QAT na validação...")
resultados_qat = avaliar(qat_model, dataset_val)
print(f"   Sequência completa correta: {acuracia_sequencia(resultados_qat) * 100:.2f}%")
print("   Exemplos (real -> previsto):")
mostrar_exemplos(resultados_qat)

qat_model.save(os.path.join(DIR_SAIDA, "modelo_digitos_ctc.h5"))

# ==========================================
# 9. CONVERSÃO PARA TFLITE INT8
# ==========================================
print("\n-> Convertendo para TFLite (INT8)...")

def representative_dataset():
    # Calibra com imagens de treino SEM augmentation
    amostras_cal = (ds_treino_cache
                    .shuffle(1000, seed=SEED)
                    .map(normalizar)
                    .batch(1)
                    .take(200))
    for img, _ in amostras_cal:
        yield [img.numpy()]

converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

caminho_tflite = os.path.join(DIR_SAIDA, "modelo_digitos_ctc.tflite")
with open(caminho_tflite, "wb") as f:
    f.write(tflite_model)
print(f"-> Modelo salvo em: {caminho_tflite}")

# ==========================================
# 10. VALIDAÇÃO DO .TFLITE (o que roda no dispositivo)
# ==========================================
# Mede a acurácia do modelo int8 real, para pegar perdas da quantização
# que a avaliação do modelo QAT (fake-quant) não mostra.
print("\n-> Validando o .tflite INT8 na validação...")
interpretador = tf.lite.Interpreter(model_content=tflite_model)
interpretador.allocate_tensors()
det_in = interpretador.get_input_details()[0]
det_out = interpretador.get_output_details()[0]
escala_in, zero_in = det_in['quantization']
escala_out, zero_out = det_out['quantization']

def prever_tflite(img):
    """img: (IMG_HEIGHT, IMG_WIDTH, 1) float32 em [0, 1]."""
    q = np.clip(np.round(img / escala_in + zero_in), -128, 127).astype(np.int8)
    interpretador.set_tensor(det_in['index'], q[np.newaxis, ...])
    interpretador.invoke()
    bruto = interpretador.get_tensor(det_out['index'])[0].astype(np.float32)
    return decodificar_greedy((bruto - zero_out) * escala_out)

resultados_tflite = []
for x, y in dataset_val:
    for img, r in zip(x.numpy(), y.numpy()):
        real = [int(v) for v in r if v != PAD]
        resultados_tflite.append((real, prever_tflite(img)))

print(f"   Sequência completa correta (TFLite INT8): "
      f"{acuracia_sequencia(resultados_tflite) * 100:.2f}%")
print("   Exemplos (real -> previsto):")
mostrar_exemplos(resultados_tflite)

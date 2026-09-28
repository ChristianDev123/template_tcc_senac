import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import random
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

# A largura reflete os 7 roletes máximos (7 * 24px)
IMG_HEIGHT = 48
IMG_WIDTH = 168
N_DIGITOS = 7
N_CLASSES = 11        # 0-9 + 'X' (posição vazia, índice 10)
BATCH_SIZE = 32

# Treino em duas fases: float primeiro, QAT depois (fine-tuning curto)
EPOCAS_FLOAT = 300
EPOCAS_QAT = 80

os.makedirs(DIR_CSVS, exist_ok=True)
os.makedirs(DIR_SAIDA, exist_ok=True)

# ==========================================
# 2. MAPEAMENTO DOS ARQUIVOS E RÓTULOS
# ==========================================
print("-> Mapeando arquivos e higienizando rótulos das pastas...")

imagens_por_rotulo = {}   # rótulo de 7 caracteres -> lista de caminhos
pastas_ignoradas = []

for root, dirs, files in os.walk(DIR_DATASET):
    nome_pasta = os.path.basename(root)

    if not nome_pasta or nome_pasta.startswith('.') or nome_pasta == "visores":
        continue

    imgs = [os.path.join(root, f) for f in files
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not imgs:
        continue

    rotulo = nome_pasta.strip().upper()

    # Medidores de 6 dígitos ganham um 'X' (posição vazia) no final
    if len(rotulo) == 6:
        rotulo += 'X'

    if len(rotulo) != N_DIGITOS or not all(c in '0123456789X' for c in rotulo):
        pastas_ignoradas.append(nome_pasta)
        continue

    imagens_por_rotulo.setdefault(rotulo, []).extend(imgs)

if not imagens_por_rotulo:
    raise ValueError(f"Nenhuma imagem válida encontrada em: {DIR_DATASET}.")

if pastas_ignoradas:
    print(f"   [AVISO] {len(pastas_ignoradas)} pasta(s) ignorada(s) por nome inválido. "
          f"Exemplos: {pastas_ignoradas[:5]}")

def rotulo_para_indices(rotulo):
    return [10 if c == 'X' else int(c) for c in rotulo]

def indices_para_texto(indices):
    return ''.join('X' if int(i) == 10 else str(int(i)) for i in indices)

# ------------------------------------------
# Split por PASTA (leitura), e não por imagem.
# Se a mesma leitura tiver várias imagens, um split por imagem coloca
# quase-duplicatas no treino e na validação e infla a métrica de validação.
# ------------------------------------------
rotulos = sorted(imagens_por_rotulo.keys())
random.Random(SEED).shuffle(rotulos)

corte = int(len(rotulos) * 0.8)
rotulos_treino, rotulos_val = rotulos[:corte], rotulos[corte:]
if not rotulos_treino or not rotulos_val:
    raise ValueError("Poucas pastas para separar treino e validação.")

def montar_listas(lista_rotulos):
    caminhos, labels = [], []
    for r in lista_rotulos:
        idx = rotulo_para_indices(r)
        for caminho in imagens_por_rotulo[r]:
            caminhos.append(caminho)
            labels.append(idx)
    return caminhos, np.array(labels, dtype=np.int32)

caminhos_treino, labels_treino = montar_listas(rotulos_treino)
caminhos_val, labels_val = montar_listas(rotulos_val)

print(f"-> {len(rotulos)} leituras distintas | treino: {len(caminhos_treino)} imagens "
      f"({len(rotulos_treino)} pastas) | validação: {len(caminhos_val)} imagens "
      f"({len(rotulos_val)} pastas)")

# ==========================================
# 3. PIPELINE tf.data
# ==========================================
def carregar(caminho, rotulos_):
    img = tf.io.read_file(caminho)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.cast(tf.round(img), tf.uint8)          # uint8 deixa o cache leve
    img = tf.reshape(img, [IMG_HEIGHT, IMG_WIDTH, 1])
    return img, rotulos_

def normalizar(img, rotulos_):
    img = tf.cast(img, tf.float32) / 255.0
    # Tupla posicional (não dict): após o quantize_model os nomes das saídas
    # mudam (d1 -> quant_d1) e um dict por nome deixaria de casar.
    return img, tuple(rotulos_[i] for i in range(N_DIGITOS))

def augmentar(img, rotulos_):
    # Apenas variações que preservam o significado dos dígitos.
    # NÃO usar flip: espelhar inverte os dígitos e a ordem das posições.
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    # Deslocamento pequeno (±3px vertical, ±4px horizontal) via pad + crop
    img = tf.pad(img, [[3, 3], [4, 4], [0, 0]], mode='SYMMETRIC')
    img = tf.image.random_crop(img, [IMG_HEIGHT, IMG_WIDTH, 1])
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, rotulos_

ds_treino_cache = (tf.data.Dataset.from_tensor_slices((caminhos_treino, labels_treino))
                   .map(carregar, num_parallel_calls=AUTOTUNE)
                   .cache())

# Treino é reembaralhado a cada época (antes, a ordem era sempre a mesma)
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
# 4. MÉTRICA REAL: SEQUÊNCIA COMPLETA CORRETA
# ==========================================
# A accuracy por cabeça engana: d1-d3 quase sempre são 0 e d7 quase sempre é X.
# O que importa na prática é acertar os 7 dígitos ao mesmo tempo.
def avaliar(modelo, dataset):
    todos_preds, todos_reais = [], []
    for x, y in dataset:
        saidas = modelo(x, training=False)
        preds = tf.stack([tf.argmax(s, axis=-1, output_type=tf.int32) for s in saidas], axis=1)
        reais = tf.stack(list(y), axis=1)
        todos_preds.append(preds.numpy())
        todos_reais.append(reais.numpy())
    return np.concatenate(todos_preds), np.concatenate(todos_reais)

class AcuraciaSequenciaCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        preds, reais = avaliar(self.model, self.dataset)
        acc = float(np.mean(np.all(preds == reais, axis=1)))
        if logs is not None:
            logs['val_seq_acc'] = acc
        print(f" - val_seq_acc: {acc:.4f}")

# ==========================================
# 5. ARQUITETURA MULTI-HEAD CNN (11 CLASSES)
# ==========================================
print("\n-> Construindo a Rede Multi-Head...")

inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.25)(x)          # Flatten -> Dense tem ~1M pesos: overfit fácil
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.25)(x)

saidas = [layers.Dense(N_CLASSES, activation='softmax', name=f'd{i + 1}')(x)
          for i in range(N_DIGITOS)]

base_model = models.Model(inputs=inputs, outputs=saidas)

# ==========================================
# 6. FASE 1 — TREINO EM PONTO FLUTUANTE
# ==========================================
print("\n-> Fase 1: treino em ponto flutuante (sem quantização)")
base_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
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
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
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
# 8. EXPORTAÇÃO DO HISTÓRICO E AVALIAÇÃO FINAL
# ==========================================
print("\n-> Exportando métricas de treino para CSV...")
df_float = pd.DataFrame(history_float.history)
df_float['fase'] = 'float'
df_qat = pd.DataFrame(history_qat.history)
df_qat['fase'] = 'qat'
df_historico = pd.concat([df_float, df_qat], ignore_index=True)
caminho_csv = os.path.join(DIR_CSVS, "historico_treino_multihead.csv")
df_historico.to_csv(caminho_csv, index_label="Epoca")
print(f"-> Arquivo CSV salvo em: {caminho_csv}")

print("\n-> Avaliação final do modelo QAT na validação...")
preds, reais = avaliar(qat_model, dataset_val)
acc_seq = float(np.mean(np.all(preds == reais, axis=1)))
acc_pos = np.mean(preds == reais, axis=0)
print(f"   Sequência completa correta: {acc_seq * 100:.2f}%")
print("   Acurácia por posição: " +
      " | ".join(f"d{i + 1}: {a * 100:.1f}%" for i, a in enumerate(acc_pos)))

rng = np.random.default_rng(SEED)
amostras = rng.choice(len(reais), size=min(10, len(reais)), replace=False)
print("   Exemplos (real -> previsto):")
for i in amostras:
    ok = "OK  " if np.array_equal(reais[i], preds[i]) else "ERRO"
    print(f"     {ok} {indices_para_texto(reais[i])} -> {indices_para_texto(preds[i])}")

qat_model.save(os.path.join(DIR_SAIDA, "modelo_multihead.h5"))

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

caminho_tflite = os.path.join(DIR_SAIDA, "modelo_multihead.tflite")
with open(caminho_tflite, "wb") as f:
    f.write(tflite_model)

print(f"-> Modelo Multi-Head salvo com sucesso em: {caminho_tflite}")
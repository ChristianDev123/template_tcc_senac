import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import math
import random
from collections import Counter
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_model_optimization as tfmot

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

DIR_DIGITOS_SOLTOS = os.path.join(DIRETORIO_ATUAL, "dataset", "num")
DIR_AMOSTRAS_SINT = os.path.join(DIRETORIO_ATUAL, "resultados_visuais", "amostras_sinteticas")

FRACAO_SINTETICA = 0.4
FRACAO_SINTETICA_QAT = 0.15     # no fine-tuning QAT o modelo vê mais dados reais
MAX_GLIFOS_POR_CLASSE = 1000    # limita a memória
GLIFO_H = 64                    # altura padrão das máscaras de dígitos soltos
N_AMOSTRAS_DEBUG = 16           # imagens sintéticas salvas para inspeção visual
SUFIXO_SAIDA = "_sint"          # não sobrescreve os modelos CTC treinados sem sintético

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
# 2. MAPEAMENTO DOS ARQUIVOS E RÓTULOS (DADOS REAIS)
# ==========================================
# Os X no FINAL do nome da pasta são descartados ("000777X" -> "000777").
# Um X no meio do nome não tem representação no CTC: a pasta é ignorada.
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

# Split por PASTA (leitura), para não vazar quase-duplicatas para a validação.
# A validação é SEMPRE 100% real: nunca entra dado sintético nela.
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
# 3. DÍGITOS SOLTOS -> MÁSCARAS
# ==========================================
def extrair_mascara(img_u8):
    """Converte um dígito solto (qualquer cor/polaridade) em uma máscara 0..1
    recortada rente ao dígito e normalizada para GLIFO_H de altura."""
    fundo = np.bincount((img_u8 >> 3).ravel()).argmax() * 8 + 4   # nível de fundo (moda)
    dif = np.abs(img_u8.astype(np.float32) - fundo)
    if dif.max() < 30:
        return None                                # imagem praticamente vazia
    mascara = dif / dif.max()
    ys, xs = np.where(mascara > 0.25)
    if len(ys) == 0:
        return None
    mascara = mascara[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = mascara.shape
    if h < 4 or w < 2:
        return None
    novo_w = max(1, int(round(w * GLIFO_H / h)))
    return cv2.resize(mascara, (novo_w, GLIFO_H), interpolation=cv2.INTER_AREA)

glifos = {d: [] for d in range(10)}
if FRACAO_SINTETICA > 0 and os.path.isdir(DIR_DIGITOS_SOLTOS):
    print("\n-> Carregando dígitos soltos...")
    for d in range(10):
        pasta = os.path.join(DIR_DIGITOS_SOLTOS, str(d))
        if not os.path.isdir(pasta):
            continue
        arquivos = sorted(f for f in os.listdir(pasta)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        random.Random(SEED + d).shuffle(arquivos)
        for f in arquivos[:MAX_GLIFOS_POR_CLASSE]:
            img = cv2.imread(os.path.join(pasta, f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            m = extrair_mascara(img)
            if m is not None:
                glifos[d].append(m)
    print("   Dígitos carregados por classe: " +
          ", ".join(f"{d}: {len(glifos[d])}" for d in range(10)))

usar_sintetico = FRACAO_SINTETICA > 0 and all(len(glifos[d]) > 0 for d in range(10))
if FRACAO_SINTETICA > 0 and not usar_sintetico:
    print(f"   [AVISO] Dígitos soltos não encontrados/incompletos em '{DIR_DIGITOS_SOLTOS}'. "
          f"Treinando apenas com dados reais.")

# ==========================================
# 4. GERADOR DE SEQUÊNCIAS SINTÉTICAS
# ==========================================
def ajustar_para_ctc(img):
    """Replica o pré-processamento dos dados reais (função `carregar`):
    redimensiona mantendo a proporção e preenche com a média da última coluna."""
    h, w = img.shape
    escala = min(IMG_HEIGHT / h, IMG_WIDTH / w)
    nova_h = max(1, int(round(h * escala)))
    nova_w = max(1, int(round(w * escala)))
    img = cv2.resize(img.astype(np.float32), (nova_w, nova_h), interpolation=cv2.INTER_LINEAR)
    fundo = float(img[:, -1:].mean())
    canvas = np.full((IMG_HEIGHT, IMG_WIDTH), fundo, dtype=np.float32)
    canvas[:nova_h, :nova_w] = img
    return np.clip(np.round(canvas), 0, 255).astype(np.uint8)

def sortear_sequencia():
    n = random.choices([5, 6, 7, 8], weights=[0.05, 0.40, 0.50, 0.05])[0]
    n = min(n, MAX_DIGITOS)
    # Leituras reais costumam começar com zeros à esquerda (000905, 0001013...)
    if random.random() < 0.6:
        k = random.randint(1, n - 1)
        return [0] * k + [random.randint(0, 9) for _ in range(n - k)]
    return [random.randint(0, 9) for _ in range(n)]

def gerar_imagem_sintetica(digitos):
    """Cola os dígitos em células de largura fixa (como roletes) sobre um fundo,
    e aplica degradações leves. Retorna uint8 (altura x largura) em tamanho nativo."""
    n = len(digitos)
    h0 = random.randint(40, 96)
    cell_w = max(8, int(h0 * random.uniform(0.45, 0.70)))
    margem = random.randint(0, 6)
    largura = n * cell_w + 2 * margem

    if random.random() < 0.5:      # fundo claro, dígito escuro
        fundo, tinta = random.uniform(0.65, 1.0), random.uniform(0.0, 0.35)
    else:                          # fundo escuro, dígito claro
        fundo, tinta = random.uniform(0.0, 0.3), random.uniform(0.65, 1.0)

    canvas = np.full((h0, largura), fundo, dtype=np.float32)
    altura_base = h0 * random.uniform(0.55, 0.85)

    for i, d in enumerate(digitos):
        m = random.choice(glifos[d])
        gh = min(h0, max(4, int(round(altura_base * random.uniform(0.95, 1.05)))))
        gw = max(2, int(round(m.shape[1] * gh / m.shape[0])))
        gw = min(gw, max(2, int(cell_w * 0.9)))    # cabe na célula (comprime só a largura)
        mascara = cv2.resize(m, (gw, gh), interpolation=cv2.INTER_AREA)

        x0 = margem + i * cell_w + (cell_w - gw) // 2 + random.randint(-2, 2)
        y0 = (h0 - gh) // 2 + random.randint(-3, 3)
        x0 = int(np.clip(x0, 0, largura - gw))
        y0 = int(np.clip(y0, 0, h0 - gh))

        regiao = canvas[y0:y0 + gh, x0:x0 + gw]
        canvas[y0:y0 + gh, x0:x0 + gw] = regiao * (1.0 - mascara) + tinta * mascara

    # Divisórias finas entre os roletes
    if random.random() < 0.5:
        cor_sep = fundo + (tinta - fundo) * random.uniform(0.2, 0.7)
        for i in range(1, n):
            x = margem + i * cell_w
            canvas[:, x:x + 1] = cor_sep

    # Iluminação desigual (gradiente horizontal ou vertical)
    if random.random() < 0.6:
        ga = random.uniform(-0.25, 0.25)
        eixo = random.choice([0, 1])
        rampa = np.linspace(1.0 - ga, 1.0 + ga, canvas.shape[eixo], dtype=np.float32)
        canvas = canvas * (rampa[None, :] if eixo == 1 else rampa[:, None])

    # Desfoque leve e ruído
    if random.random() < 0.6:
        canvas = cv2.GaussianBlur(canvas, (0, 0), random.uniform(0.3, 1.5))
    canvas = canvas + np.random.normal(0, random.uniform(0.0, 0.05), canvas.shape).astype(np.float32)

    return (np.clip(canvas, 0.0, 1.0) * 255).astype(np.uint8)

def gerador_sintetico():
    while True:
        digitos = sortear_sequencia()
        img = ajustar_para_ctc(gerar_imagem_sintetica(digitos))
        rotulo = np.array(digitos + [PAD] * (MAX_DIGITOS - len(digitos)), dtype=np.int32)
        yield img[..., np.newaxis], rotulo

if usar_sintetico:
    os.makedirs(DIR_AMOSTRAS_SINT, exist_ok=True)
    gen_debug = gerador_sintetico()
    for i in range(N_AMOSTRAS_DEBUG):
        img_dbg, rot_dbg = next(gen_debug)
        texto = ''.join(str(v) for v in rot_dbg if v != PAD)
        cv2.imwrite(os.path.join(DIR_AMOSTRAS_SINT, f"sint_{i:02d}_{texto}.png"), img_dbg[..., 0])
    print(f"-> {N_AMOSTRAS_DEBUG} amostras sintéticas salvas em: {DIR_AMOSTRAS_SINT}")

# ==========================================
# 5. PIPELINE tf.data
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

assinatura_sintetica = (tf.TensorSpec((IMG_HEIGHT, IMG_WIDTH, 1), tf.uint8),
                        tf.TensorSpec((MAX_DIGITOS,), tf.int32))

def montar_dataset_treino(fracao_sintetica):
    """Mistura dados reais (reembaralhados a cada passada) com sequências
    sintéticas na proporção pedida. O dataset é infinito: use steps_per_epoch."""
    assert 0.0 <= fracao_sintetica < 1.0
    ds_real = ds_treino_cache.shuffle(len(caminhos_treino), seed=SEED).repeat()

    if fracao_sintetica > 0:
        ds_sint = tf.data.Dataset.from_generator(gerador_sintetico,
                                                 output_signature=assinatura_sintetica)
        ds = tf.data.Dataset.sample_from_datasets(
            [ds_real, ds_sint], weights=[1.0 - fracao_sintetica, fracao_sintetica], seed=SEED)
    else:
        ds = ds_real

    ds = (ds.map(normalizar, num_parallel_calls=AUTOTUNE)
            .map(augmentar, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))

    # Uma "época" continua equivalendo a ~1 passada pelos dados REAIS
    passos = math.ceil(len(caminhos_treino) / (BATCH_SIZE * (1.0 - fracao_sintetica)))
    return ds, passos

dataset_val = (tf.data.Dataset.from_tensor_slices((caminhos_val, labels_val))
               .map(carregar, num_parallel_calls=AUTOTUNE)
               .cache()
               .map(normalizar, num_parallel_calls=AUTOTUNE)
               .batch(BATCH_SIZE)
               .prefetch(AUTOTUNE))

# ==========================================
# 6. LOSS CTC, DECODIFICAÇÃO E MÉTRICA
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
        print(f" - val_seq_acc (real): {acc:.4f}")

# ==========================================
# 7. ARQUITETURA CRNN SEM RECORRÊNCIA (CONV -> CTC)
# ==========================================
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
# 8. FASE 1 — TREINO EM PONTO FLUTUANTE
# ==========================================
# É normal a loss CTC ficar "travada" por várias épocas (a rede começa
# prevendo só blank) e val_seq_acc ficar em 0 até ela "descolar".
frac_float = FRACAO_SINTETICA if usar_sintetico else 0.0
dataset_treino_float, passos_float = montar_dataset_treino(frac_float)
print(f"\n-> Fase 1: treino em ponto flutuante "
      f"({frac_float * 100:.0f}% sintético, {passos_float} passos/época)")

base_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
    loss=ctc_loss
)

history_float = base_model.fit(
    dataset_treino_float,
    validation_data=dataset_val,
    epochs=EPOCAS_FLOAT,
    steps_per_epoch=passos_float,
    callbacks=[
        AcuraciaSequenciaCallback(dataset_val),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                             patience=10, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30,
                                         restore_best_weights=True)
    ]
)

# ==========================================
# 9. FASE 2 — FINE-TUNING COM QAT
# ==========================================
frac_qat = FRACAO_SINTETICA_QAT if usar_sintetico else 0.0
dataset_treino_qat, passos_qat = montar_dataset_treino(frac_qat)
print(f"\n-> Fase 2: QAT com learning rate baixa "
      f"({frac_qat * 100:.0f}% sintético, {passos_qat} passos/época)")

qat_model = tfmot.quantization.keras.quantize_model(base_model)
qat_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0),
    loss=ctc_loss
)

history_qat = qat_model.fit(
    dataset_treino_qat,
    validation_data=dataset_val,
    epochs=EPOCAS_QAT,
    steps_per_epoch=passos_qat,
    callbacks=[
        AcuraciaSequenciaCallback(dataset_val),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15,
                                         restore_best_weights=True)
    ]
)

# ==========================================
# 10. EXPORTAÇÃO DO HISTÓRICO E AVALIAÇÃO DO MODELO QAT
# ==========================================
print("\n-> Exportando métricas de treino para CSV...")
df_float = pd.DataFrame(history_float.history)
df_float['fase'] = 'float'
df_qat = pd.DataFrame(history_qat.history)
df_qat['fase'] = 'qat'
df_historico = pd.concat([df_float, df_qat], ignore_index=True)
caminho_csv = os.path.join(DIR_CSVS, f"historico_treino_ctc{SUFIXO_SAIDA}.csv")
df_historico.to_csv(caminho_csv, index_label="Epoca")
print(f"-> Arquivo CSV salvo em: {caminho_csv}")

def mostrar_exemplos(resultados, n=10):
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(resultados), size=min(n, len(resultados)), replace=False)
    for i in idx:
        real, pred = resultados[i]
        ok = "OK  " if real == pred else "ERRO"
        print(f"     {ok} {lista_para_texto(real)} -> {lista_para_texto(pred)}")

print("\n-> Avaliação do modelo QAT na validação (imagens reais)...")
resultados_qat = avaliar(qat_model, dataset_val)
print(f"   Sequência completa correta: {acuracia_sequencia(resultados_qat) * 100:.2f}%")
print("   Exemplos (real -> previsto):")
mostrar_exemplos(resultados_qat)

qat_model.save(os.path.join(DIR_SAIDA, f"modelo_digitos_ctc{SUFIXO_SAIDA}.h5"))

# ==========================================
# 11. CONVERSÃO PARA TFLITE INT8
# ==========================================
print("\n-> Convertendo para TFLite (INT8)...")

def representative_dataset():
    # Calibra com imagens REAIS de treino, sem augmentation
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

caminho_tflite = os.path.join(DIR_SAIDA, f"modelo_digitos_ctc{SUFIXO_SAIDA}.tflite")
with open(caminho_tflite, "wb") as f:
    f.write(tflite_model)
print(f"-> Modelo salvo em: {caminho_tflite}")

# ==========================================
# 12. VALIDAÇÃO DO .TFLITE (o que roda no dispositivo)
# ==========================================
print("\n-> Validando o .tflite INT8 na validação (imagens reais)...")
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
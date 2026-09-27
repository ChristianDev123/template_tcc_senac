import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import pandas as pd # <- Nova importação
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_model_optimization as tfmot

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
DIR_DATASET = os.path.join(DIRETORIO_ATUAL, "dataset", "visores")
DIR_CSVS = os.path.join(DIRETORIO_ATUAL, "CSVs") # <- Novo diretório

# A largura reflete agora os 7 roletes máximos (7 * 24px)
IMG_HEIGHT = 48
IMG_WIDTH = 168

os.makedirs(DIR_CSVS, exist_ok=True) # <- Garante que a pasta existe

# ==========================================
# 2. PIPELINE DE DADOS
# ==========================================
print("-> Mapeando ficheiros e higienizando rótulos das pastas...")

arquivos_img = []
r_d1, r_d2, r_d3, r_d4, r_d5, r_d6, r_d7 = [], [], [], [], [], [], []

for root, dirs, files in os.walk(DIR_DATASET):
    nome_pasta = os.path.basename(root)
    
    if not nome_pasta or nome_pasta.startswith('.') or nome_pasta == "visores":
        continue
        
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg')):
            rotulo = nome_pasta.strip().upper()
            
            if len(rotulo) == 6:
                rotulo += 'X'
                
            if len(rotulo) != 7:
                continue
                
            try:
                valores = [10 if char == 'X' else int(char) for char in rotulo]
                
                r_d1.append(valores[0])
                r_d2.append(valores[1])
                r_d3.append(valores[2])
                r_d4.append(valores[3])
                r_d5.append(valores[4])
                r_d6.append(valores[5])
                r_d7.append(valores[6])
                arquivos_img.append(os.path.join(root, file))
            except ValueError:
                continue

if not arquivos_img:
    raise ValueError(f"Nenhuma imagem válida encontrada em: {DIR_DATASET}.")

print(f"-> Sucesso! {len(arquivos_img)} imagens prontas para treino.")

def processar_imagem(caminho, d1, d2, d3, d4, d5, d6, d7):
    img = tf.io.read_file(caminho)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.reshape(img, [IMG_HEIGHT, IMG_WIDTH, 1])
    
    return img, (d1, d2, d3, d4, d5, d6, d7)

dataset = tf.data.Dataset.from_tensor_slices((
    arquivos_img, r_d1, r_d2, r_d3, r_d4, r_d5, r_d6, r_d7
))

dataset = dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=False)
dataset = dataset.map(processar_imagem, num_parallel_calls=tf.data.AUTOTUNE)

tamanho_dataset = len(arquivos_img)
tamanho_treino = int(tamanho_dataset * 0.8)

dataset_treino = dataset.take(tamanho_treino).batch(32).prefetch(tf.data.AUTOTUNE)
dataset_val = dataset.skip(tamanho_treino).batch(32).prefetch(tf.data.AUTOTUNE)

# ==========================================
# 3. ARQUITETURA MULTI-HEAD CNN (11 CLASSES)
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
x = layers.Dense(128, activation='relu')(x)

out1 = layers.Dense(11, activation='softmax', name='d1')(x)
out2 = layers.Dense(11, activation='softmax', name='d2')(x)
out3 = layers.Dense(11, activation='softmax', name='d3')(x)
out4 = layers.Dense(11, activation='softmax', name='d4')(x)
out5 = layers.Dense(11, activation='softmax', name='d5')(x)
out6 = layers.Dense(11, activation='softmax', name='d6')(x)
out7 = layers.Dense(11, activation='softmax', name='d7')(x)

base_model = models.Model(inputs=inputs, outputs=[out1, out2, out3, out4, out5, out6, out7])

qat_model = tfmot.quantization.keras.quantize_model(base_model)
qat_model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

# ==========================================
# 4. TREINAMENTO E EXPORTAÇÃO CSV
# ==========================================
print("\n-> Iniciando Treinamento QAT...")
# Captura o histórico do treino numa variável
history = qat_model.fit(
    dataset_treino,
    validation_data=dataset_val,
    epochs=1000,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)]
)

# Salva o histórico num ficheiro CSV
print("\n-> Exportando métricas de treino para CSV...")
df_historico = pd.DataFrame(history.history)
caminho_csv = os.path.join(DIR_CSVS, "historico_treino_multihead.csv")
df_historico.to_csv(caminho_csv, index_label="Epoca")
print(f"-> Arquivo CSV salvo em: {caminho_csv}")

DIR_SAIDA = os.path.join(DIRETORIO_ATUAL, "Modelos")
os.makedirs(DIR_SAIDA, exist_ok=True)
qat_model.save(os.path.join(DIR_SAIDA, "modelo_multihead.h5"))

# ==========================================
# 5. CONVERSÃO PARA TFLITE INT8
# ==========================================
print("\n-> Convertendo para TFLite (INT8)...")

def representative_dataset():
    for img, _ in dataset_treino.unbatch().batch(1).take(150):
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
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import load_img, img_to_array
import tensorflow_model_optimization as tfmot # Importação do módulo QAT

# ==========================================
# 1. CONFIGURAÇÕES
# ========================================== 
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))

DIR_IMG = os.path.join(DIRETORIO_ATUAL, "dataset", "images")
DIR_MASK = os.path.join(DIRETORIO_ATUAL, "dataset", "masks")
IMG_SIZE = (192, 192)

# ==========================================
# 2. CARREGAMENTO DOS DADOS E SEPARAÇÃO
# ==========================================
print("-> Carregando imagens e máscaras...")
arquivos = sorted(os.listdir(DIR_IMG))
X, Y = [], []

for arquivo in arquivos:
    if not arquivo.endswith(('.png', '.jpg', '.jpeg')): continue
    
    # Carrega Imagem e Máscara
    caminho_img = os.path.join(DIR_IMG, arquivo)
    caminho_mask = os.path.join(DIR_MASK, arquivo)
    
    img_array = img_to_array(load_img(caminho_img, color_mode="grayscale", target_size=IMG_SIZE)) / 255.0
    mask_array = img_to_array(load_img(caminho_mask, color_mode="grayscale", target_size=IMG_SIZE)) / 255.0
    
    # Binariza a máscara
    mask_array = np.where(mask_array > 0.5, 1.0, 0.0)
    
    X.append(img_array)
    Y.append(mask_array)

X_train_full = np.array(X)
Y_train_full = np.array(Y)
print(f"Dataset carregado! Total: {len(X_train_full)} amostras.")

# Separação manual para treino (80%) e validação (20%)
split_idx = int(len(X_train_full) * 0.8)
X_train, X_val = X_train_full[:split_idx], X_train_full[split_idx:]
Y_train, Y_val = Y_train_full[:split_idx], Y_train_full[split_idx:]

# ==========================================
# 3. DATA AUGMENTATION UNIFICADO
# ==========================================
def augment(image, mask):
    # O flip horizontal afeta tanto a imagem quanto a máscara simultaneamente
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    return image, mask

# Dataset de treino com Augmentation e Otimização de Performance
dataset_treino = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
dataset_treino = dataset_treino.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
dataset_treino = dataset_treino.batch(1).prefetch(tf.data.AUTOTUNE)

# Dataset de validação não recebe augmentation, apenas batching
dataset_val = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
dataset_val = dataset_val.batch(1).prefetch(tf.data.AUTOTUNE)

# ==========================================
# 4. ARQUITETURA DO MODELO E QAT
# ==========================================
print("\n-> Criando modelo...")
inputs = layers.Input(batch_shape=(1, IMG_SIZE[0], IMG_SIZE[1], 1))

c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
p1 = layers.MaxPooling2D((2, 2))(c1)

c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
p2 = layers.MaxPooling2D((2, 2))(c2)

u1 = layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(p2)
u2 = layers.Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same', activation='relu')(u1)

outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(u2)

base_model = models.Model(inputs=[inputs], outputs=[outputs])

# Envolve a arquitetura para o Quantization-Aware Training
qat_model = tfmot.quantization.keras.quantize_model(base_model)
qat_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ==========================================
# 5. CALLBACKS DE PROTEÇÃO
# ==========================================
class TempoEstimadoCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.tempo_inicio = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epocas_feitas = epoch + 1
        epocas_totais = self.params['epochs']
        epocas_restantes = epocas_totais - epocas_feitas
        
        tempo_decorrido = time.time() - self.tempo_inicio
        tempo_medio_por_epoca = tempo_decorrido / epocas_feitas
        tempo_restante_seg = tempo_medio_por_epoca * epocas_restantes
        
        h = int(tempo_restante_seg // 3600)
        m = int((tempo_restante_seg % 3600) // 60)
        s = int(tempo_restante_seg % 60)
        
        if epocas_restantes > 0:
            print(f"\n[ETA Global] Faltam {epocas_restantes} épocas. Fim estimado em: {h}h {m}m {s}s\n")

# Lista de Callbacks integrando CSV, Checkpoint de Backup e Early Stopping
callbacks_treino = [
    TempoEstimadoCallback(),
    tf.keras.callbacks.CSVLogger('log_treinamento.csv', append=True),
    tf.keras.callbacks.ModelCheckpoint('backup_medidor.keras', save_best_only=True, monitor='val_loss'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
]

# ==========================================
# 6. TREINAMENTO
# ==========================================
print("\n-> Iniciando Treinamento com QAT")
# Usa o qat_model e os datasets gerados pelo tf.data
qat_model.fit(
    dataset_treino, 
    validation_data=dataset_val, 
    epochs=10000, 
    callbacks=callbacks_treino
)

qat_model.save("modelo_medidor_qat.h5") 

# ==========================================
# 7. CONVERSÃO E QUANTIZAÇÃO (TFLITE INT8)
# ==========================================
print("\n-> Iniciando Conversão")

def representative_dataset():
    # Extrai amostras do dataset otimizado para calibrar as ativações
    for amostra, _ in dataset_treino.take(100):
        yield [amostra.numpy()]

# Converte diretamente do modelo treinado com QAT
converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# Restrições de compatibilidade para microcontroladores
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()

with open("modelo_esp32.tflite", "wb") as f:
    f.write(tflite_quant_model)

print("\n->Arquivo 'modelo_esp32.tflite' gerado.")
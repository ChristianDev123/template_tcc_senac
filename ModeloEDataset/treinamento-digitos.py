import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_model_optimization as tfmot

# ==========================================
# 1. ARQUITETURA DA CNN DE DÍGITOS
# ==========================================
def criar_modelo_digitos():
    inputs = layers.Input(shape=(8, 8, 1))
    
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)
    
    outputs = layers.Dense(10, activation='softmax')(x)
    
    return models.Model(inputs, outputs)

base_model = criar_modelo_digitos()

# ==========================================
# 2. QUANTIZATION-AWARE TRAINING (QAT)
# ==========================================
qat_model = tfmot.quantization.keras.quantize_model(base_model)

qat_model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# ==========================================
# 3. CARREGAMENTO DO DATASET E TREINAMENTO
# ==========================================
DIR_DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "num")
BATCH_SIZE = 32
IMG_SIZE = (8, 8)

dataset_treino_bruto = tf.keras.utils.image_dataset_from_directory(
    DIR_DATASET,
    validation_split=0.2,
    subset="training",
    seed=123,
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

dataset_val_bruto = tf.keras.utils.image_dataset_from_directory(
    DIR_DATASET,
    validation_split=0.2,
    subset="validation",
    seed=123,
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

normalization_layer = layers.Rescaling(1./255)

dataset_treino = dataset_treino_bruto.map(
    lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

dataset_val = dataset_val_bruto.map(
    lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

qat_model.fit(
    dataset_treino, 
    validation_data=dataset_val, 
    epochs=100
)

DIR_SAIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Modelos")
os.makedirs(DIR_SAIDA, exist_ok=True)

caminho_h5 = os.path.join(DIR_SAIDA, "modelo_digitos_qat.h5")
qat_model.save(caminho_h5)

# ==========================================
# 4. CONVERSÃO PARA TFLITE (INT8)
# ==========================================
def representative_dataset():
    for imagens, _ in dataset_treino.take(1):
        for i in range(imagens.shape[0]):
            yield [np.expand_dims(imagens[i].numpy(), axis=0)]

converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()

caminho_tflite = os.path.join(DIR_SAIDA, "classificador_digitos.tflite")
with open(caminho_tflite, "wb") as f:
    f.write(tflite_quant_model)

print(f"\n->Arquivos salvos na pasta: {DIR_SAIDA}")
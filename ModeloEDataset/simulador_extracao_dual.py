import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.utils import load_img, img_to_array
import tensorflow.keras.backend as K

# ==========================================
# 0. FUNÇÕES CUSTOMIZADAS DA U-NET
# ==========================================
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)
    return bce + dice_loss(y_true, y_pred)

# ==========================================
# 1. CONFIGURAÇÕES E DIRETÓRIOS
# ==========================================
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))

DIR_MODELOS = os.path.join(DIRETORIO_ATUAL, "Modelos")
DIR_IMG_REAIS = os.path.join(DIRETORIO_ATUAL, "dataset", "imagens_reais")
DIR_CSVS = os.path.join(DIRETORIO_ATUAL, "CSVs")
ARQUIVO_TXT = os.path.join(DIR_CSVS, "leituras_finais.txt")
ARQUIVO_CSV = os.path.join(DIR_CSVS, "leituras_finais.csv")

DIR_RECORTES = os.path.join(DIRETORIO_ATUAL, "resultados_visuais", "recortes_dual")

os.makedirs(DIR_CSVS, exist_ok=True)
os.makedirs(DIR_RECORTES, exist_ok=True)

# ==========================================
# 2. CARREGAMENTO DOS DOIS MODELOS
# ==========================================
print("-> Carregando Redes Neurais (U-Net e CNN Multi-Head)...")
custom_objects = {
    'dice_coef': dice_coef,
    'dice_loss': dice_loss,
    'bce_dice_loss': bce_dice_loss
}

with tfmot.quantization.keras.quantize_scope():
    # Carrega a U-Net injetando a matemática customizada
    unet_model = tf.keras.models.load_model(os.path.join(DIR_MODELOS, "modelo_medidor.h5"), custom_objects=custom_objects)
    
    # Carrega o modelo Multi-Head treinado (certifique-se de que o nome do ficheiro está correto)
    digit_model = tf.keras.models.load_model(os.path.join(DIR_MODELOS, "modelo_multihead.h5"))

# ==========================================
# 3. FUNÇÕES DE PROCESSAMENTO
# ==========================================
def extrair_coordenadas_unet(caminho_img):
    img_pil = load_img(caminho_img, color_mode="grayscale", target_size=(384, 384))
    img_array = img_to_array(img_pil) / 255.0
    img_input = np.expand_dims(img_array, axis=0)
    
    predicao = unet_model.predict(img_input, verbose=0)[0]
    pred_binaria = np.where(predicao > 0.5, 1.0, 0.0)
    
    mask_uint8 = (pred_binaria.squeeze() * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask_corrigida = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    
    contornos, _ = cv2.findContours(mask_corrigida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contornos: return None
        
    maior_contorno = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(maior_contorno) < 80: return None
        
    return cv2.boundingRect(maior_contorno)

def classificar_visor_multihead(recorte_visor, nome_arquivo):
    # Salva o recorte bruto para debug
    cv2.imwrite(os.path.join(DIR_RECORTES, f"visor_original_{nome_arquivo}"), recorte_visor)
    
    # 1. Redimensiona a imagem inteira para as dimensões exatas que a Multi-Head exige
    # Largura: 168px, Altura: 48px
    visor_resized = cv2.resize(recorte_visor, (168, 48), interpolation=cv2.INTER_AREA)
    
    # 2. Equaliza o contraste
    visor_norm = cv2.normalize(visor_resized, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(DIR_RECORTES, f"visor_rede_{nome_arquivo}"), visor_norm)
    
    # 3. Prepara o tensor (1, 48, 168, 1)
    tensor_input = visor_norm.astype(np.float32) / 255.0
    tensor_input = np.expand_dims(tensor_input, axis=(0, -1))
    
    # 4. Inferência - A rede devolve 7 arrays (as 7 cabeças), cada um com 11 probabilidades
    predicoes = digit_model.predict(tensor_input, verbose=0)
    
    leitura_final = ""
    
    # Avalia a predição de cada um dos 7 roletes
    for pred in predicoes:
        digito = np.argmax(pred[0])
        # A classe 10 é o "X" (vazio/borda).
        if digito != 10: 
            leitura_final += str(digito)
            
    return leitura_final

# ==========================================
# 4. PIPELINE DE EXECUÇÃO
# ==========================================
print("\n-> Iniciando Extração (Pipeline Dual Multi-Head)...\n")

dados_csv = []

with open(ARQUIVO_TXT, "w", encoding="utf-8") as arquivo_txt:
    arquivo_txt.write("Arquivo | Leitura (Dual Model)\n")
    arquivo_txt.write("-" * 40 + "\n")

    arquivos = sorted(os.listdir(DIR_IMG_REAIS))
    
    for arquivo in arquivos:
        if not arquivo.endswith(('.png', '.jpg', '.jpeg')): continue
        
        caminho_img = os.path.join(DIR_IMG_REAIS, arquivo)
        bbox = extrair_coordenadas_unet(caminho_img)
        
        if not bbox:
            linha = f"{arquivo}: FALHA_LOCALIZACAO"
            print(linha)
            arquivo_txt.write(linha + "\n")
            dados_csv.append({"Arquivo": arquivo, "Leitura": "FALHA_LOCALIZACAO"})
            continue
            
        x_unet, y_unet, w_unet, h_unet = bbox
        
        img_original = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)
        h_orig, w_orig = img_original.shape
        
        # Mapeia as coordenadas da U-Net (384x384) de volta para o tamanho real da foto
        fator_x = w_orig / 384.0
        fator_y = h_orig / 384.0
        
        x_real = int(x_unet * fator_x)
        y_real = int(y_unet * fator_y)
        w_real = int(w_unet * fator_x)
        h_real = int(h_unet * fator_y)
        
        recorte_visor = img_original[y_real:y_real+h_real, x_real:x_real+w_real]
        
        # Chama a nova função adaptada para o modelo de 7 cabeças
        leitura = classificar_visor_multihead(recorte_visor, arquivo)
        
        linha = f"{arquivo}: {leitura}"
        print(linha)
        arquivo_txt.write(linha + "\n")
        dados_csv.append({"Arquivo": arquivo, "Leitura": leitura})

# Exportação do CSV
df_resultados = pd.DataFrame(dados_csv)
df_resultados.to_csv(ARQUIVO_CSV, index=False)

print(f"\n-> Leituras exportadas para TXT: '{ARQUIVO_TXT}'")
print(f"-> Leituras exportadas para CSV: '{ARQUIVO_CSV}'")
print(f"-> Imagens de depuração salvas em: '{DIR_RECORTES}'")
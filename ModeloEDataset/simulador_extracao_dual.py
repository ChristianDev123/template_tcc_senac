import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.utils import load_img, img_to_array

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
# 2. CARREGAMENTO DOS DOIS MODELOS (Dual TinyML)
# ==========================================
print("-> Carregando Redes Neurais (U-Net e CNN de Dígitos)...")
with tfmot.quantization.keras.quantize_scope():
    unet_model = tf.keras.models.load_model(os.path.join(DIR_MODELOS, "modelo_medidor.h5"))
    digit_model = tf.keras.models.load_model(os.path.join(DIR_MODELOS, "modelo_digitos.h5"))

# ==========================================
# 3. FUNÇÕES DE PROCESSAMENTO
# ==========================================
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

def pad_and_resize(img_slice, target_size=48):
    """Protege as proporções originais colando o recorte num fundo quadrado."""
    h, w = img_slice.shape
    if h == 0 or w == 0:
        return np.zeros((target_size, target_size), dtype=np.uint8)
        
    max_dim = max(h, w)
    top = (max_dim - h) // 2
    bottom = max_dim - h - top
    left = (max_dim - w) // 2
    right = max_dim - w - left
    
    cor_fundo = int(np.median(img_slice))
    img_quadrada = cv2.copyMakeBorder(img_slice, top, bottom, left, right, cv2.BORDER_CONSTANT, value=cor_fundo)
    
    return cv2.resize(img_quadrada, (target_size, target_size), interpolation=cv2.INTER_AREA)

def classificar_fatias(recorte_visor, nome_arquivo):
    altura, largura = recorte_visor.shape
    largura_fatia = largura // 6
    leitura_final = ""
    
    cv2.imwrite(os.path.join(DIR_RECORTES, f"visor_{nome_arquivo}"), recorte_visor)
    
    for i in range(6):
        # 1. Corte matemático simples e previsível
        inicio_x = i * largura_fatia
        fim_x = (i + 1) * largura_fatia if i < 5 else largura
        
        fatia_img = recorte_visor[:, inicio_x:fim_x]
        
        # 2. Elimina a linha de plástico nas extremidades do rolete
        margem = int(fatia_img.shape[1] * 0.12)
        if margem > 0 and fatia_img.shape[1] > margem * 2:
            fatia_img = fatia_img[:, margem:-margem]
        
        # 3. Redimensiona para o novo modelo de 48x48 protegendo a proporção
        fatia_48x48 = pad_and_resize(fatia_img, 48)
        
        # 4. Melhora o contraste (resolve o problema dos números desbotados)
        fatia_48x48 = cv2.normalize(fatia_48x48, None, 0, 255, cv2.NORM_MINMAX)
        
        # (Opcional) Se a CNN aprendeu com fundo preto e números brancos, descomente abaixo:
        # fatia_48x48 = cv2.bitwise_not(fatia_48x48)
        
        cv2.imwrite(os.path.join(DIR_RECORTES, f"fatia_{i}_{nome_arquivo}"), fatia_48x48)
        
        # 5. Inferência
        fatia_normalizada = fatia_48x48.astype(np.float32) / 255.0
        fatia_tensor = np.expand_dims(fatia_normalizada, axis=(0, -1))
        
        predicao = digit_model.predict(fatia_tensor, verbose=0)[0]
        digito = np.argmax(predicao)
        
        leitura_final += str(digito)
        
    return leitura_final

# ==========================================
# 4. PIPELINE DE EXECUÇÃO
# ==========================================
print("\n-> Iniciando Extração (Pipeline Dual)...\n")

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
        
        fator_x = w_orig / 384.0
        fator_y = h_orig / 384.0
        
        x_real = int(x_unet * fator_x)
        y_real = int(y_unet * fator_y)
        w_real = int(w_unet * fator_x)
        h_real = int(h_unet * fator_y)
        
        recorte_visor = img_original[y_real:y_real+h_real, x_real:x_real+w_real]
        
        leitura = classificar_fatias(recorte_visor, arquivo)
        
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
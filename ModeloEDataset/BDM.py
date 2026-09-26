import os
os.environ['TF_USE_LEGACY_KERAS'] = '1' 

import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.utils import load_img, img_to_array

# ==========================================
# 1. CONFIGURAÇÕES E DIRETÓRIOS
# ==========================================
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))

DIR_MODELO = os.path.join(DIRETORIO_ATUAL, "Modelos", "modelo_medidor_qat.h5")
DIR_IMG_TESTE = os.path.join(DIRETORIO_ATUAL, "dataset", "test_images")
DIR_MASK_TESTE = os.path.join(DIRETORIO_ATUAL, "dataset", "test_masks")
DIR_IMG_REAIS = os.path.join(DIRETORIO_ATUAL, "dataset", "imagens_reais")
DIR_RESULTADOS = os.path.join(DIRETORIO_ATUAL, "resultados_visuais")
DIR_CSVS = os.path.join(DIRETORIO_ATUAL, "CSVs")

IMG_SIZE = (192, 192)

os.makedirs(DIR_RESULTADOS, exist_ok=True)
os.makedirs(DIR_IMG_REAIS, exist_ok=True)
os.makedirs(DIR_CSVS, exist_ok=True)

# ==========================================
# 2. FUNÇÕES DE MÉTRICAS E VALIDAÇÃO
# ==========================================
def calcular_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return (intersection / union) * 100 if union > 0 else 100.0

def calcular_dice(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    return (2. * intersection / (y_true.sum() + y_pred.sum())) * 100 if (y_true.sum() + y_pred.sum()) > 0 else 100.0

def validar_hidrometro(predicao_binaria):

    # Converte a matriz binária (0.0 e 1.0) para formato de imagem de 8 bits (0 e 255)
    mask_uint8 = (predicao_binaria.squeeze() * 255).astype(np.uint8)
    
    # Cria um elemento estruturante (um retângulo de 15x15 pixels). 
    # Esse bloco varre a imagem conectando pixels laranjas que estejam próximos.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask_corrigida = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    
    # Dilatação extra se a máscara ainda estiver muito falha
    # mask_corrigida = cv2.dilate(mask_corrigida, kernel, iterations=1)
    
    # Encontra os contornos na máscara já fundida e corrigida
    contornos, _ = cv2.findContours(mask_corrigida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contornos:
        return "Inexistente", None

    # Foca apenas no maior objeto detectado
    maior_contorno = max(contornos, key=cv2.contourArea)
    area_contorno = cv2.contourArea(maior_contorno)
    
    if area_contorno < 80: 
        return "Inexistente (Ou Ruído)", None
        
    # Obtém as coordenadas da caixa delimitadora
    x, y, w, h = cv2.boundingRect(maior_contorno)
    proporcao = w / float(h)
    
    if w < 40 or h < 15:
        status = "Mal Posicionado (Muito Distante)"
    elif 1.5 <= proporcao <= 7.0:
        status = "Identificável"
    else:
        status = "Mal Posicionado (Ângulo/Corte Incorreto)"
        
    return status, (x, y, w, h)

# ==========================================
# 3. CARREGAMENTO DO MODELO QAT
# ==========================================
print("-> Carregando modelo QAT...")
with tfmot.quantization.keras.quantize_scope():
    model = tf.keras.models.load_model(DIR_MODELO)

# ==========================================
# 4. LOOP DE TESTE UNIFICADO
# ==========================================
print("-> Iniciando inferência e validação...")
dados_tabela = []

pastas_para_testar = [
    {"img_dir": DIR_IMG_TESTE, "mask_dir": DIR_MASK_TESTE, "tipo": "Dataset Teste"},
    {"img_dir": DIR_IMG_REAIS, "mask_dir": None, "tipo": "Mundo Real"}
]

for config in pastas_para_testar:
    if not os.path.exists(config["img_dir"]): continue
    
    arquivos = sorted(os.listdir(config["img_dir"]))
    
    for arquivo in arquivos:
        if not arquivo.endswith(('.png', '.jpg', '.jpeg')): continue
        
        caminho_img = os.path.join(config["img_dir"], arquivo)
        caminho_mask = os.path.join(config["mask_dir"], arquivo) if config["mask_dir"] else None
        
        img_pil = load_img(caminho_img, color_mode="grayscale", target_size=IMG_SIZE)
        img_array = img_to_array(img_pil) / 255.0
        img_input = np.expand_dims(img_array, axis=0)
        
        inicio = time.perf_counter()
        predicao = model.predict(img_input, verbose=0)[0]
        fim = time.perf_counter()
        tempo_ms = (fim - inicio) * 1000
        
        pred_binaria = np.where(predicao > 0.5, 1.0, 0.0)
        area_pixels = np.sum(pred_binaria)
        
        iou, dice = None, None
        status_deteccao = "N/A"
        
        tem_mascara = caminho_mask and os.path.exists(caminho_mask)
        
        if tem_mascara:
            # Fluxo 1: Com Gabarito (Apenas Métricas)
            mask_array = img_to_array(load_img(caminho_mask, color_mode="grayscale", target_size=IMG_SIZE)) / 255.0
            mask_binaria = np.where(mask_array > 0.5, 1.0, 0.0)
            
            iou = calcular_iou(mask_binaria, pred_binaria)
            dice = calcular_dice(mask_binaria, pred_binaria)
            status_deteccao = "Gabarito Validado"
            
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(img_array.squeeze(), cmap='gray')
            axs[0].set_title('Original')
            axs[1].imshow(mask_binaria.squeeze(), cmap='gray')
            axs[1].set_title('Gabarito')
            axs[2].imshow(pred_binaria.squeeze(), cmap='jet', alpha=0.7)
            axs[2].set_title(f'Predição (IoU: {iou:.1f}%)')
            
        else:
            # Fluxo 2: Mundo Real (Validação Estrutural)
            status_deteccao, bbox = validar_hidrometro(pred_binaria)
            
            # Desenha um retângulo vermelho na imagem original se encontrar o visor
            img_display = cv2.cvtColor((img_array.squeeze() * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(img_display, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(img_display)
            axs[0].set_title(f'Status: {status_deteccao}')
            axs[1].imshow(pred_binaria.squeeze(), cmap='jet', alpha=0.7)
            axs[1].set_title(f'Predição ({int(area_pixels)} px)')

        for ax in axs: ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(DIR_RESULTADOS, f"{config['tipo'].replace(' ', '_')}_{arquivo}"))
        plt.close(fig)

        dados_tabela.append({
            "Fonte": config["tipo"],
            "Arquivo": arquivo,
            "Veredito": status_deteccao,
            "Latência (ms)": round(tempo_ms, 2),
            "Área (px)": int(area_pixels),
            "IoU (%)": round(iou, 2) if iou else "N/A",
            "Dice (%)": round(dice, 2) if dice else "N/A"
        })

# ==========================================
# 5. EXPORTAÇÃO DOS DADOS
# ==========================================
df_resultados = pd.DataFrame(dados_tabela)

# Salva o arquivo dentro da nova pasta configurada
caminho_csv = os.path.join(DIR_CSVS, "tabela_provas_funcionamento.csv")
df_resultados.to_csv(caminho_csv, index=False)

print("\n--- RESULTADOS DO TESTE E VALIDAÇÃO ---")
print(df_resultados.to_markdown(index=False))
print(f"\n-> Arquivo de dados salvo em: '{caminho_csv}'")
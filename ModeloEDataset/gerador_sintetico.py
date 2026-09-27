import os
import cv2
import numpy as np
import random

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
DIR_DIGITOS_ANTIGOS = os.path.join(DIRETORIO_ATUAL, "dataset", "num")
DIR_VISORES_NOVOS = os.path.join(DIRETORIO_ATUAL, "dataset", "visores")

TOTAL_GERAR = 1500
ALTURA_DIGITO = 48
LARGURA_DIGITO = 24

LARGURA_MAXIMA = 168 

# ==========================================
# 2. CARREGAMENTO DOS NÚMEROS ISOLADOS
# ==========================================
banco_imagens = {str(i): [] for i in range(10)}

for i in range(10):
    pasta_digito = os.path.join(DIR_DIGITOS_ANTIGOS, str(i))
    if os.path.exists(pasta_digito):
        banco_imagens[str(i)] = [os.path.join(pasta_digito, f) for f in os.listdir(pasta_digito) if f.endswith(('.png', '.jpg', '.jpeg'))]

for i in range(10):
    if len(banco_imagens[str(i)]) == 0:
        raise ValueError(f"A pasta do número {i} está vazia ou não existe em {DIR_DIGITOS_ANTIGOS}!")

print(f"-> Gerando {TOTAL_GERAR} visores (misturando 6 e 7 dígitos)...")

# ==========================================
# 3. MONTAGEM DOS VISORES "FRANKENSTEIN"
# ==========================================
for _ in range(TOTAL_GERAR):
    # Sorteia se este hidrômetro específico terá 6 ou 7 números
    qtd_digitos = random.choice([6, 7])
    sequencia = "".join([str(random.randint(0, 9)) for _ in range(qtd_digitos)])
    
    fatias = []
    for num in sequencia:
        caminho_escolhido = random.choice(banco_imagens[num])
        img = cv2.imread(caminho_escolhido, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (LARGURA_DIGITO, ALTURA_DIGITO))
        
        if random.random() > 0.3:
            cv2.line(img, (0, 0), (0, ALTURA_DIGITO), (80), 1)
            
        fatias.append(img)
    
    # Junta as fatias. Se tiver 6 dígitos = 144px. Se 7 = 168px.
    visor_base = np.hstack(fatias)
    
    if qtd_digitos == 6:
        # Preenche a largura que falta com fundo cinza para não esticar os números
        falta_largura = LARGURA_MAXIMA - visor_base.shape[1]
        pad_esq = falta_largura // 2
        pad_dir = falta_largura - pad_esq
        
        cor_fundo = int(np.median(visor_base))
        visor_completo = cv2.copyMakeBorder(visor_base, 0, 0, pad_esq, pad_dir, cv2.BORDER_CONSTANT, value=cor_fundo)
        
        # Adiciona o Token 'X' para o nome da pasta ter sempre 7 caracteres (ex: "830015X")
        nome_pasta = sequencia + "X"
    else:
        visor_completo = visor_base
        nome_pasta = sequencia
        
    pasta_destino = os.path.join(DIR_VISORES_NOVOS, nome_pasta)
    os.makedirs(pasta_destino, exist_ok=True)
    
    hash_aleatorio = random.randint(10000, 99999)
    caminho_salvar = os.path.join(pasta_destino, f"sintetico_{hash_aleatorio}.jpg")
    cv2.imwrite(caminho_salvar, visor_completo)

print(f"\n-> Sucesso! Imagens geradas na pasta: {DIR_VISORES_NOVOS}")
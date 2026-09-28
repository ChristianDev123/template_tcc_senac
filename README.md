# Leitor de Hidrómetro com TinyML (Dual-Model Vision)

O sistema utiliza uma abordagem *Dual-Model* (dois modelos a correr em sequência):
1. **Modelo Detetor (U-Net):** Segmenta a imagem para localizar a região exata do visor do hidrómetro.
2. **Modelo Classificador (CNN Multi-Head):** Recebe o recorte do visor e classifica os 6 dígitos numéricos simultaneamente.

Toda a inferência é convertida para `INT8` (Quantization-Aware Training) para garantir que cabe na SRAM limitada do hardware alvo, possuindo também um simulador C++ nativo a correr em WSL (Linux) para validar a alocação de memória antes do *deploy* no hardware físico.

---

## 📁 Estrutura do Repositório

Abaixo encontra-se a organização de diretórios e ficheiros do projeto:

### 📂 Diretórios Principais
*   **`dataset/`**: Contém todos os dados utilizados no treino e validação.
    *   `images/` e `masks/`: Imagens completas dos hidrómetros e as respetivas máscaras binárias para treinar a U-Net.
    *   `visores/` e `num/`: Recortes específicos dos visores e dígitos individuais usados para treinar as redes de leitura.
    *   `imagens_reais/`: Fotografias do mundo real recolhidas para testar o pipeline em condições adversas fora do dataset de treino.
    *   `Crédito.txt`: Informações e origem da base de dados utilizada.
*   **`Modelos/`**: Diretório de saída para os modelos gerados pelo Keras (`.h5`) e os modelos quantizados convertidos (`.tflite`).
*   **`CSVs/`**: Armazena os históricos de treino gerados pelo Pandas (evolução da *loss* e *accuracy*) e tabelas de validação de inferência.
*   **`resultados_visuais/`**: Imagens exportadas pelos scripts de teste, exibindo as predições de segmentação, *bounding boxes* e recortes gerados.
*   **`tflite-micro/`**: Pasta do ambiente Linux (WSL) contendo o núcleo matemático compilado (`.a`) do Google TensorFlow Lite Micro.
*   **`Backups/`** e **`ModeloBrutoAntigoParaTeste/`**: Ficheiros e arquiteturas de versões anteriores retidos para controlo de qualidade e comparação de métricas.

### 📄 Scripts e Código-Fonte
*   **`treinamento-detector.py`**: Treina a rede U-Net com métricas de *Dice Loss* para localizar o visor na imagem.
*   **`treinamento-digitos-multihead.py`**: Treina a CNN com 7 cabeças de saída independentes para ler todos os dígitos de uma só vez.
*   **`treinamento-digitos-ctc.py`**: Script de treino experimental utilizando *Connectionist Temporal Classification* (CTC) para a leitura.
*   **`treinamento-digitos-ctc-sint.py`**: Script de treino experimental utilizanbdo CTC assim como o anterior mas com dados sintéticos para a leitura.
*   **`gerador_sintetico.py`**: Pipeline de pré-processamento e *Data Augmentation* para gerar variações artificiais que enriquecem o dataset de leitura e extração de digitos.
*   **`simulador_extracao_dual.py`**: Script Python que simula a pipeline completa: carrega uma imagem, usa a U-Net para recortar a região de interesse, trata a imagem, e injeta na CNN para devolver a leitura final.
*   **`BDM.py`**: Script unificado de validação e extração de métricas de IoU (Intersection over Union) e Dice para os modelos tanto em imagens do dataset quanto em imagens externas.
*   **`BDMwsl.cpp`**: Simulador nativo escrito em C++ puro. Simula as restrições de memória de um microcontrolador através da limitação da *Tensor Arena*, executa as operações *TfLite*, processa matrizes de imagem manualmente e serve de ponte antes de gravar o código no ESP32.
*   **`diagnostico_modelo.py`**: Script para teste da area lida pelo modelo de recorte.
*   **`modelo_medidor.h`**: *Dump* em matriz hexadecimal do modelo TFLite (`unsigned char array`), pronto a ser embutido em código C/C++.
*   **`requirements.txt`**: Ficheiro com as dependências do ambiente Python necessárias para executar os scripts.

---

### Como Preparar o Ambiente
Crie um ambiente virtual e instale as dependências:
```bash
python -m venv .venv
source .venv/bin/activate  #ou apenas .venv/bin/activate no windows
pip install -r requirements.txt
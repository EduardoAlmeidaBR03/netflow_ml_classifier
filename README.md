# Classificador ML de Fluxos de rede - TCC

Este projeto implementa uma abordagem de classificação de fluxos de rede utilizando Machine Learning, desenvolvida como Trabalho de Conclusão de Curso (TCC). Inicialmente, realiza-se o mapeamento entre as aplicações identificadas via Deep Packet Inspection (DPI) e os fluxos NetFlow correspondentes, classificando os fluxos para compor o dataset. Em seguida, os dados são processados para gerar novas features relevantes, o conjunto é balanceado para reduzir vieses e, por fim, um modelo de Decision Tree é treinado para a classificação final do tráfego de rede.

## Visão Geral do Projeto

O projeto está dividido em **duas fases principais**:

**Fase 1-3: Criação e Processamento dos Dados**
- Extração e filtragem de dados DPI
- Processamento e cálculo de características
- Balanceamento e categorização final

**Fase 4: Aprendizado de Máquina**
- Treinamento do modelo Árvore de Decisão
- Avaliação e otimização de desempenho

##  Estrutura do Projeto

```
netflow_ml_classifier_tcc/
├── 1-classificador_de_fluxos_com_DPI/     # Criação de Dados: Análise DPI e filtragem
├── 2-processando_dataset/                 # Processamento: Engenharia de características
├── balanceado_dataset/                    # Finalização: Balanceamento dos dados
├── treinando_modelo/                      # Aprendizado de Máquina: Treinamento do modelo
├── requirements.txt                       # Dependências do projeto
└── README.md                             # Este arquivo
```

##  Descrição dos Módulos

> **Etapas do Fluxo de Trabalho:**
> - **Módulos 1-3**: Criação e processamento completo dos dados
> - **Módulo 4**: Treinamento e avaliação do modelo de Aprendizado de Máquina

### 1. Classificador de Fluxos com DPI (`1-classificador_de_fluxos_com_DPI/`) 
* Etapa 1: Criação dos Dados - Filtragem e Extração*

**Script:** `DPI_flow_analyzer.py`

**Funcionalidade:** Analisa arquivos de Deep Packet Inspection (DPI) para extrair informações de IPs e portas, e filtra fluxos de rede por aplicação específica.

**O que faz:**
- Extrai IPs e portas de origem/destino dos arquivos DPI (.txt)
- Filtra fluxos NetFlow baseado nos critérios extraídos
- Gera arquivos CSV filtrados por aplicação
- Suporta múltiplas aplicações simultaneamente

**Pastas:**
- `dpi_txt/`: Contém arquivos TXT com dados DPI de entrada
- `flows/`: Contém arquivos CSV com fluxos NetFlow originais
- `porta_ip/`: Armazena critérios extraídos (IPs e portas por aplicação)
- `flows_filtrados/`: Fluxos filtrados por aplicação específica

### 2. Processamento de Dataset (`2-processando_dataset/`)
* Etapa 2: Processamento dos Dados - Engenharia de Características*

**Script:** `processing.py`

**Funcionalidade:** Processa conjuntos de dados de fluxos NetFlow calculando características derivadas.

**O que faz:**
- Calcula métricas derivadas como velocidade de pacotes, razões entre bytes
- Realiza divisão segura evitando erros de divisão por zero
- Processa dados em partes para otimizar memória
- Trata valores infinitos e NaN
- Reorganiza colunas em formato padronizado

**Arquivos:**
- `dataset.csv`: Conjunto de dados original de entrada
- `dataset_tratado.csv`: Conjunto de dados processado com características calculadas

### 3. Dataset Balanceado (`balanceado_dataset/`)
* Etapa 3: Finalização dos Dados - Balanceamento e Categorização*

**Script:** `create_balanced_dataset.py`

**Funcionalidade:** Cria um conjunto de dados balanceado categorizando protocolos e aplicando técnicas de amostragem.

**O que faz:**
- Categoriza protocolos em grupos funcionais:
  - STREAMING (Spotify, YouTube, Netflix, etc.)
  - CONFERENCE (Teams, Discord, Skype, etc.)
  - FILE_TRANSFER (FTP, GoogleCloud, etc.)
  - SOCIAL (Twitter, Instagram, Facebook)
  - GENERAL (HTTP, DNS, SSH, etc.)
- Balanceia classes usando amostragem com ruído controlado
- Remove categorias "OTHER" para manter consistência
- Gera conjunto de dados final com número igual de amostras por classe

**Arquivos:**
- `dataset_tratado_balanceado.csv`: Conjunto de dados final balanceado

### 4. Treinamento do Modelo (`treinando_modelo/`)
* Etapa 4: Aprendizado de Máquina - Treinamento e Avaliação*

**Script:** `decision_tree_final_optimized.py`

**Funcionalidade:** Treina e avalia um modelo Árvore de Decisão otimizado para classificação de protocolos de rede.

**O que faz:**
- Carrega conjunto de dados balanceado
- Treina Árvore de Decisão com parâmetros otimizados
- Avalia modelo com múltiplas métricas (acurácia, pontuação F1, precisão, revocação)
- Salva modelo treinado em formato joblib
- Gera relatório completo de classificação

**Arquivos:**
- `decision_tree_model.joblib`: Modelo treinado salvo

##  Como Executar

### Pré-requisitos

1. **Python**
2. **Instalar dependências:**
```bash
pip install -r requirements.txt
```

### Dependências (requirements.txt)

```
pandas       # Manipulação de dados
numpy        # Computação numérica
scikit-learn # Aprendizado de Máquina
joblib       # Serialização de modelos
```

### Execução dos Scripts

#### 1. Análise DPI e Filtragem de Fluxos
```bash
# Executar para uma aplicação específica
python3 1-classificador_de_fluxos_com_DPI/DPI_flow_analyzer.py YouTube

# Executar para múltiplas aplicações
python3 1-classificador_de_fluxos_com_DPI/DPI_flow_analyzer.py WhatsApp Teams Skype Discord YouTube

# Aplicações suportadas: WhatsApp, Teams, Skype, Discord, YouTube, GoogleCloud, etc.
```

#### 2. Processamento dos Dados
```bash
python3 2-processando_dataset/processing.py
```

#### 3. Criação do Conjunto de Dados Balanceado
```bash
python3 balanceado_dataset/create_balanced_dataset.py
```

#### 4. Treinamento do Modelo
```bash
python3 treinando_modelo/decision_tree_final_optimized.py
```

##  Fluxo de Trabalho Completo

###  **Fase 1-3: Criação e Processamento dos Dados**
1. **Coleta de Dados** → Arquivos .txt em `dpi_txt/`
2. **Análise** → Extração de IPs/portas específicos por aplicação
3. **Filtragem de Fluxos** → Aplicação de critérios aos fluxos NetFlow
4. **Processamento** → Cálculo de características e normalização
5. **Balanceamento** → Criação de conjunto de dados equilibrado por categorias

###  **Fase 4: Aprendizado de Máquina**
6. **Treinamento** → Modelo Árvore de Decisão otimizado
7. **Avaliação** → Métricas de desempenho e modelo final



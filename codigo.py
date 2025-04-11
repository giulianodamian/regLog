#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nome do Programa: codigo.py
Descrição: Aplica o modelo de regressão logística para um determinado banco de dados.
Autor: Giuliano Damian
Data de Criação: [DD/MM/AAAA]
Versão: 0.1.0
"""

######################################################################
'''Importação das bibliotecas '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
#from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
import warnings
#from typing import Tuple, List, Dict, Optional, Union
import sys
from pathlib import Path

# Configurações globais
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)


######################################################################
'''Variáveis de Entrada'''
class Config:
    INPUT_FILE = 'bank.csv'
    OUTPUT_TXT ='resultado_reg_log.txt'
    MAX_ITERATION = 10000



######################################################################
'''Função de registro de informações no .txt'''
def save_to_txt(content, file_path, mode='a'):
    """Salva conteúdo em arquivo TXT."""
    with open(file_path, mode, encoding='utf-8') as f:
        f.write(content + "\n")

#abre o .txt no início
with open(Config.OUTPUT_TXT, 'w') as f:
        f.write('RELATÓRIO DE ANÁLISE ESTATÍSTICA\n')
        f.write('=' * 50 + '\n\n')

######################################################################
'''Carregamento e Manipulação Inicial dos Dados'''
df = pd.read_csv(Config.INPUT_FILE, sep = ',')

df.columns = df.columns.str.strip().str.lower()
if df['y'].dtype == 'object':
            df['y'] = df['y'].str.lower().map({'yes': 1, 'no': 0, 'sim': 1, 'não': 0,'1': 1, '0': 0,'true': 1, 'false': 0}).fillna(0)

df['y'] = pd.to_numeric(df['y'], errors = 'coerce').fillna(0).astype(np.int8)
df['education'] = df['education'].replace({
            'basic.9y': 'basic', 'basic.6y': 'basic', 'basic.4y': 'basic',
            'university.degree': 'university', 'professional.course': 'professional',
            'high.school': 'high_school', 'illiterate': 'illiterate', 'unknown': 'unknown'
        }).str.lower()
cat_vars = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]

df = pd.get_dummies(df, columns= cat_vars, drop_first=True, dtype = np.int8)
with open('Dados_Corrigidos.txt', 'w') as f:
        f.write(df.to_string(index=False))

X = df.drop('y', axis = 1)
y = df['y'].values

x = X.apply(pd.to_numeric, errors = 'coerce').fillna(0)
X = X.astype(np.float32)

if X.isnull().sum().sum()>0:
       raise ValueError('Há valores nulos nas features após o tratamento!')
if not set(np.unique(y)).issubset({0, 1}):
       raise ValueError('Variável dependente deve conter apenas valores de 0 e 1!')

X_var = X.columns.tolist()

######################################################################

'''Análise Univariada dos Dados'''

save_to_txt("\n=== ANÁLISE UNIVARIADA ===", Config.OUTPUT_TXT)

try:
    X = sm.add_constant(X)
    model = sm.Logit(y, X).fit(disp=0, method='bfgs')
except Exception as e:
    raise ValueError("Erro na regressão logística: {str(e)}")

univariable_results = []

for feature in X_var:
    X_uni = sm.add_constant(X[[feature]])
    model = sm.Logit(y, X_uni).fit(disp=0, method='bfgs')

    univariable_results.append({
        "Variable": feature,
        "Coefficient": model.params[1],
        "Standard Error": model.bse[1],
        "P-value": model.pvalues[1]
    })

    # save_to_txt(f"\n--- {feature} ---", Config.OUTPUT_TXT)
    # save_to_txt(model.summary().as_text(), Config.OUTPUT_TXT)

df_results = pd.DataFrame(univariable_results)
save_to_txt("\nResultados:", Config.OUTPUT_TXT)
save_to_txt(df_results.to_string(index=False),Config.OUTPUT_TXT)

######################################################################
'''Verificação da linearidade dos Dados'''

######################################################################
'''Remoção das Variáveis Altamente Correlacionadas'''

######################################################################
'''Remoção de Variáveis com alto VIF'''

######################################################################
'''Realização da seleção de variáveis stepwise para construção de um 
modelo Multivariado'''

######################################################################
'''Avaliação do Modelo'''

######################################################################
'''Aplicação do RFECV para modelagem e seleção automática final
 dos recursos'''

######################################################################

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 22:24:33 2024

@author: Saulo Klein
"""

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import numpy as np

class ValidacaoCruzada:
    
    def __init__(self, classificador, processador, n_splits=5):
        self.classificador = classificador
        self.processador = processador
        self.n_splits = n_splits
        self.validar()
        
    def validar(self):
        kfold = StratifiedKFold(
            n_splits=self.n_splits, 
            shuffle=True, 
            random_state=1)

        acuracias = []
        matrizes = []
        metricas = []
        
        for indice_treinamento, indice_teste in kfold.split(self.processador.previsores, np.zeros(shape=(self.processador.previsores.shape[0], 1))):
            # Treinamento com o classificador específico
            self.classificador.treinarModelo(self.processador.previsores[indice_treinamento], self.processador.classe.iloc[indice_treinamento, 0])

            # Previsão
            previsoes = self.classificador.preverModelo(self.processador.previsores[indice_teste])

            # Avaliação
            acuracia = accuracy_score(self.processador.classe.iloc[indice_teste, 0], previsoes)
            
            metricas.append(precision_recall_fscore_support(self.processador.classe.iloc[indice_teste, 0], previsoes))
            matrizes.append(confusion_matrix(self.processador.classe.iloc[indice_teste, 0], previsoes))
            acuracias.append(acuracia)
        
        self.matriz_media = np.mean(matrizes, axis=0)
        self.matriz_desvio_padrao = np.std(matrizes, axis=0)
        self.acuracias = np.asarray(acuracias)
        self.acuracia_final_media = np.mean(acuracias)
        self.acuracia_final_desvio_padrao = np.std(acuracias)
        self.metricas_medias = np.mean(metricas, axis=0)
        self.metricas_desvio_padrao = np.std(metricas, axis=0)
        
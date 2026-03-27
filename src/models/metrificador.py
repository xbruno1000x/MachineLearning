# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 04:22:38 2024

@author: Saulo Klein
"""


from sklearn.metrics import confusion_matrix as cm, accuracy_score as ac
from models.classificador import Classificador

class Metrificador:

    def __init__(self, classificador: Classificador):
        self.classes = classificador.classe_teste
        self.previsoes = classificador.previsoes
        
    def matrizConfusao(self):
        return cm(self.classes, self.previsoes)
        
    def acuracia(self):
        return ac(self.classes, self.previsoes)
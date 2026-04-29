# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 22:50:59 2024

@author: Saulo Klein
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np

class Classificador:
    
    def __init__(self, processador):
        self.cols_previsores = processador.cols_previsores
        self.previsores_treinamento = processador.previsores_treinamento
        self.previsores_teste = processador.previsores_teste
        self.classe_treinamento = processador.classe_treinamento
        self.classe_teste = processador.classe_teste
        
    def KNN(self, n):
        # Geração do KNN
        self.classificador = KNeighborsClassifier(n_neighbors = n, 
                                                  metric = 'minkowski', 
                                                  p = 2)
        
        # Treinamento
        self.treinarModelo(self.previsores_treinamento, self.classe_treinamento.iloc[:, 0])
        
        # Teste
        self.previsoes = self.preverModelo(self.previsores_teste)
        
    def NaiveBayes(self):
        # Geração do Naive Bayes
        self.classificador = GaussianNB()
        
        # Treinamento
        self.treinarModelo(self.previsores_treinamento, self.classe_treinamento.iloc[:, 0])

        # Teste
        self.previsoes = self.preverModelo(self.previsores_teste)
        
    def ArvoreDecisao(self, criterion, max_depth, random_state):
        # Geração da Árvore de Decisão
        self.classificador = DecisionTreeClassifier(criterion = criterion, 
                                                    max_depth = max_depth, 
                                                    random_state = random_state)
        
        # Treinamento
        self.treinarModelo(self.previsores_treinamento, self.classe_treinamento.iloc[:, 0])

        # Teste
        self.previsoes = self.preverModelo(self.previsores_teste)
        
    
    def RandomForest(self, criterion, max_depth, n_estimators, max_features, random_state, montar_grafico=True):
        # Geração do Random Forest
        self.classificador = RandomForestClassifier(criterion = criterion, 
                                                    max_depth = max_depth, 
                                                    n_estimators = n_estimators, 
                                                    max_features = max_features, 
                                                    random_state = random_state)
        
        # Treinamento
        self.treinarModelo(self.previsores_treinamento, self.classe_treinamento.iloc[:, 0])

        # Teste
        self.previsoes = self.preverModelo(self.previsores_teste)
        if (montar_grafico):
            n_features = len(self.cols_previsores)
            plt.barh(range(n_features), self.classificador.feature_importances_, align='center')
            plt.yticks(np.arange(n_features), self.cols_previsores)
            plt.xlabel("Feature importance")
            plt.ylabel("Feature")
    
    def SVM(self, kernel, gamma, C, random_state):
        # Geração do Random Forest
        self.classificador = SVC(kernel = kernel, 
                                 gamma = gamma, 
                                 C = C, 
                                 random_state = random_state)
        
        # Treinamento
        self.treinarModelo(self.previsores_treinamento, self.classe_treinamento.iloc[:, 0])

        # Teste
        self.previsoes = self.preverModelo(self.previsores_teste)
        
        
    def RedesNeurais(self, verbose, max_iter, tol, solver, hls, activation, random_state):
        # Geração de Redes Neurais
        self.classificador = MLPClassifier(verbose = verbose,
                              max_iter= max_iter,
                              tol = tol,
                              solver = solver,
                              hidden_layer_sizes = hls,
                              activation = activation,
                              random_state = random_state)
        
        # Treinamento
        self.treinarModelo(self.previsores_treinamento, self.classe_treinamento.iloc[:, 0])
        
        # Teste
        self.previsoes = self.preverModelo(self.previsores_teste)
        
        # Plot
        plt.imshow(self.classificador.coefs_[0], interpolation='none', cmap='viridis')
        plt.yticks(range(len(self.cols_previsores)), self.cols_previsores)
        plt.xlabel("Neuronios da primeira camada")
        plt.ylabel("Característica")
        plt.colorbar()        
        
    def treinarModelo(self, previsores, classe):
        self.classificador.fit(previsores, classe)
        
    def preverModelo(self, previsores):
        return self.classificador.predict(previsores)
# -*- coding: utf-8 -*-
"""
Pre-processador para o dataset de risco de empréstimo.

@author: Bruno Faria
Created on Wed Mar 26 19:03:00 2026
"""

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns


class PreProcessador:
	scaler = StandardScaler()

	def __init__(self, configs):
		self._normalizar_configs(configs)
		self.prepararBase(self.configs)
		self.separarPrevisoresDaClasse(self.configs["col_classe"])
		self.preProcessar(self.configs)

	def _normalizar_configs(self, configs):
		"""Garante chaves opcionais com valores padrão."""
		self.configs = {
			"nome_arquivo": configs.get("nome_arquivo"),
			"col_classe": configs.get("col_classe", "LoanApproved"),
			"remover_colunas": configs.get("remover_colunas", []),
			"concatenacao": configs.get("concatenacao", None),
			"cols_categoria_ordinal": configs.get("cols_categoria_ordinal", []),
			"cols_dummy": configs.get("cols_dummy", []),
			"padronizacao": configs.get("padronizacao", False),
			"test_size": configs.get("test_size", 0.25),
			"random_state": configs.get("random_state", 0),
		}

		if not self.configs["nome_arquivo"]:
			raise ValueError("Informe o caminho em configs['nome_arquivo'].")

	def prepararBase(self, configs):
		self.lerCsv(configs["nome_arquivo"])

		for col in configs["remover_colunas"]:
			if col in self.base.columns:
				self.removerColuna(col)

		if configs["concatenacao"] is not None:
			c = configs["concatenacao"]
			self.concatenarColunas(
				c["col1"], c["col2"], c["concat"], c["col_nova"], c["drop_cols"]
			)

		self.tratarDadosAusentes()
		self.resumo = self.resumoBase()

	def lerCsv(self, nome_arquivo):
		self.base = pd.read_csv(nome_arquivo)

	def resumoBase(self):
		return self.base.describe(include="all")

	def removerColuna(self, nome_coluna):
		self.base.drop(nome_coluna, axis=1, inplace=True)

	def concatenarColunas(self, col1, col2, concatenador, nome_col_nova, remover_cols):
		concatenacao = self.base[col1].astype(str) + concatenador + self.base[col2].astype(str)
		self.base.insert(2, nome_col_nova, concatenacao, True)

		if remover_cols:
			self.removerColuna(col1)
			self.removerColuna(col2)

	def tratarDadosAusentes(self):
		"""Preenche valores ausentes por tipo de variável."""
		cols_numericas = self.base.select_dtypes(include=["number"]).columns
		cols_categoricas = self.base.select_dtypes(exclude=["number"]).columns

		for col in cols_numericas:
			mediana = self.base[col].median()
			self.base[col] = self.base[col].fillna(mediana)

		for col in cols_categoricas:
			moda = self.base[col].mode(dropna=True)
			valor_moda = moda.iloc[0] if not moda.empty else "Desconhecido"
			self.base[col] = self.base[col].fillna(valor_moda)

	def separarPrevisoresDaClasse(self, coluna_classe):
		if coluna_classe not in self.base.columns:
			raise ValueError(f"A coluna de classe '{coluna_classe}' nao existe na base.")

		colunas = self.base.columns
		self.cols_previsores = colunas.drop(coluna_classe).tolist()
		self.cols_classe = [coluna_classe]

		self.previsores = self.base[self.cols_previsores].copy()
		self.classe = self.base[self.cols_classe].copy()

	def transformarVariavelCategoriaOrdinal(self, col):
		nome = col.get("nome", "")
		ordem = col.get("ordem", None)
		if nome not in self.previsores.columns:
			return

		labelEncoder = LabelEncoder()
		if ordem is not None:
			labelEncoder.classes_ = np.array(ordem)

		# Garantir que os valores sejam convertidos para string antes da transformação
		self.previsores[nome] = self.previsores[nome].astype(str)
		self.previsores[nome] = labelEncoder.fit_transform(self.previsores[nome])

		# Garantir que o tipo final seja int
		self.previsores[nome] = self.previsores[nome].astype(int)

	def transformarVariavelDummy(self, nome_col):
		if nome_col not in self.previsores.columns:
			return

		# Garantir que os valores sejam convertidos para string antes de criar dummies
		self.previsores[nome_col] = self.previsores[nome_col].astype(str)
		dummies = pd.get_dummies(self.previsores[nome_col], prefix=nome_col, dtype=int)
		self.previsores = self.previsores.join(dummies)
		self.previsores.drop(nome_col, axis=1, inplace=True)
		self.cols_previsores = self.previsores.columns.tolist()

	def padronizarDados(self):
		valores_escalados = self.scaler.fit_transform(self.previsores)
		self.previsores = pd.DataFrame(
			valores_escalados,
			columns=self.cols_previsores,
			index=self.base.index,		)

	def preProcessar(self, configs):
		for col in configs["cols_categoria_ordinal"]:
			self.transformarVariavelCategoriaOrdinal(col)

		for col in configs["cols_dummy"]:
			self.transformarVariavelDummy(col)

		if configs["padronizacao"]:
			self.padronizarDados()

		(
			self.previsores_treinamento,
			self.previsores_teste,
			self.classe_treinamento,
			self.classe_teste,
		) = train_test_split(
			self.previsores,
			self.classe,
			test_size=configs["test_size"],
			random_state=configs["random_state"],
		)

	def correlacao(self):
		base = pd.concat([self.previsores, self.classe], axis=1)
		correlacao = base.corr()
		return sns.heatmap(
			correlacao,
			vmin=-1, vmax=1, center=0,
			cmap=sns.diverging_palette(20, 220, n=200),
			square=True,
			xticklabels=True,
			yticklabels=True
		)
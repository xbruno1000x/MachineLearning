# %%
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from models.pre_processador import PreProcessador
from models.classificador import Classificador
from models.metrificador import Metrificador
from models.validacao_cruzada import ValidacaoCruzada

csv_path = SRC_DIR / "data" / "loan_risk_balanceado.csv"

# Ative apenas um bloco de configs por vez.

# configs = {
# 	"nome_arquivo": csv_path,
# 	"cols_dummy": [],
# 	"cols_categoria_ordinal": [
# 		{
# 			"nome": "EmploymentType",
# 			"ordem": ["Self-Employed", "Salaried", "Unemployed"],
# 		},
# 		{
# 			"nome": "Education",
# 			"ordem": ["PhD", "Masters", "Bachelors", "High School"],
# 		},
# 		{
# 			"nome": "Gender",
# 		},
# 		{
# 			"nome": "City",
# 		},
# 	],
# 	"padronizacao": True,
# 	"random_state": 0,
# }

configs = {
	"nome_arquivo": csv_path,
	"cols_dummy": ["City", "Gender", "Education", "EmploymentType"],
	"cols_categoria_ordinal": [],
	"padronizacao": True,
	"random_state": 0,
}

# configs = {
# 	"nome_arquivo": csv_path,
# 	"cols_dummy": ["City"],
# 	"cols_categoria_ordinal": [
# 		{
# 			"nome": "EmploymentType",
# 			"ordem": ["Self-Employed", "Salaried", "Unemployed"],
# 		},
# 		{
# 			"nome": "Education",
# 			"ordem": ["PhD", "Masters", "Bachelors", "High School"],
# 		},
# 		{
# 			"nome": "Gender",
# 		},
# 	],
# 	"padronizacao": True,
# 	"random_state": 0,
# }

# configs = {
# 	"nome_arquivo": csv_path,
# 	"remover_colunas": ["Education", "YearsExperience", "Gender", "City"],
# 	"cols_dummy": ["EmploymentType"],
# 	"cols_categoria_ordinal": [],
# 	"padronizacao": False,
# 	"random_state": 0,
# }

# configs = {
# 	"nome_arquivo": csv_path,
# 	"remover_colunas": ["Education", "YearsExperience"],
# 	"cols_dummy": ["EmploymentType"],
# 	"cols_categoria_ordinal": [
# 		{
# 			"nome": "Gender",
# 		},
# 		{
# 			"nome": "City",
# 		},
# 	],
# 	"padronizacao": True,
# 	"random_state": 0,
# }

processador = PreProcessador(configs)

classificador = Classificador(processador)
classificador.RedesNeurais(
	verbose=False,
	max_iter=2500,
	tol=1e-7,
	solver="sgd",
	hls=[10, 10],
	activation="relu",
	random_state=0
)

metrificador = Metrificador(classificador)
acuracia = metrificador.acuracia()
matriz_confusao = metrificador.matrizConfusao()

print(f"Acuracia: {acuracia}")
print(f"Matriz de Confusao:\n{matriz_confusao}")
print(f"Iteracoes: {classificador.classificador.n_iter_}")
print(f"Loss final: {classificador.classificador.loss_:.6f}")

# %%

validacao_cruzada = ValidacaoCruzada(classificador, processador, n_splits=5)
(
	matriz_media,
	matriz_desvio_padrao,
	acuracia_final_media,
	acuracia_final_desvio_padrao,
	metricas_medias,
	metricas_desvio_padrao,
) = validacao_cruzada.metricas()

print(matriz_media)
print(matriz_desvio_padrao)
print(acuracia_final_media)
print(acuracia_final_desvio_padrao)
print(metricas_medias)
print(metricas_desvio_padrao)
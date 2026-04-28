# %%
from pathlib import Path
import pandas as pd
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

# csv_path = Path(str(SRC_DIR) + '/data/loan_risk_balanceado.csv')
csv_path = Path(str(SRC_DIR) + '/data/loan_risk_balanceado.csv')

# EmploymentType -> Self-Employed, Salaried, Unemployed
# Education -> PhD, Masters, Bachelors, High School
# City -> Chicago, San Francisco, New York, Houston

# %%
configs = {
	"random_state": 0,
    "nome_arquivo": csv_path,
	"remover_colunas": ["Education", "YearsExperience", "Gender", "City"],
    "cols_dummy": ["EmploymentType"],
    # "padronizacao": "True",
    "cols_categoria_ordinal": [
        # {
        #     "nome": "EmploymentType",
        #     "ordem": ["Self-Employed", "Salaried", "Unemployed"]
        # },
        # {
        #     "nome": "Education",
        #     "ordem": ["PhD", "Masters", "Bachelors", "High School"]
        # },
        # {
        #     "nome": "Gender"
        # },
        # {
        #     "nome": "City"
        # },
    ]
}

# %%
from models.classificador import Classificador
from models.pre_processador import PreProcessador
from models.metrificador import Metrificador

pre_processador = PreProcessador(configs)
classificador = Classificador(pre_processador)

classificador.NaiveBayes()

metrificador = Metrificador(classificador)
acuracia = metrificador.acuracia()
matriz_confusao = metrificador.matrizConfusao()

print(f"Acuracia: {round(acuracia, 6)}")
print(f"Matriz de Confusao:\n{matriz_confusao}")

# %%

pre_processador.correlacao()

# %%
from models.validacao_cruzada import ValidacaoCruzada
validacao_cruzada = ValidacaoCruzada(classificador, pre_processador, n_splits=5)
matriz_media, matriz_desvio_padrao, acuracia_final_media, acuracia_final_desvio_padrao, metricas_medias, metricas_desvio_padrao = validacao_cruzada.metricas()

print(matriz_media)
print(matriz_desvio_padrao)
print(acuracia_final_media)
print(acuracia_final_desvio_padrao)
print(metricas_medias)
print(metricas_desvio_padrao)
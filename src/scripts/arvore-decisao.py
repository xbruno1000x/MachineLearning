# %%
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

csv_path = Path(str(SRC_DIR) + '/data/loan_risk_balanceado.csv')
# csv_path = Path(str(SRC_DIR) + '/data/loan_risk_prediction_dataset.csv')

# EmploymentType -> Self-Employed, Salaried, Unemployed
# Education -> PhD, Masters, Bachelors, High School
# City -> Chicago, San Francisco, New York, Houston

# %%
# criterion = "entropy" 
criterion = "gini" 
# criterion = "log_loss" 
configs = {
    "nome_arquivo": csv_path,
    # "cols_dummy": ['City', 'Gender', 'EmploymentType', 'Education'],
    # "cols_dummy": ["City", "Gender"],
    # "cols_dummy": ["EmploymentType", "Education"],
    "padronizacao": "True",
    "cols_categoria_ordinal": [
        {
            "nome": "EmploymentType",
            "ordem": ["Self-Employed", "Salaried", "Unemployed"]
        },
        {
            "nome": "Education",
            "ordem": ["PhD", "Masters", "Bachelors", "High School"]
        },
        {
            "nome": "Gender"
        },
        {
            "nome": "City"
        },
    ]
}

from models.classificador import Classificador
from models.pre_processador import PreProcessador
from models.metrificador import Metrificador
from models.validacao_cruzada import ValidacaoCruzada
import matplotlib.pyplot as plt

configs["random_state"] = 0
pre_processador = PreProcessador(configs)
classificador = Classificador(pre_processador)

acuracias_treinamento = []
acuracias_teste = []
range_depth = range(1, 25)
for depth in range_depth:
    classificador.ArvoreDecisao(criterion=criterion, max_depth=depth, random_state=configs["random_state"])
    metrificador = Metrificador(classificador)
    acuracias_treinamento.append(classificador.classificador.score(classificador.previsores_treinamento, classificador.classe_treinamento))
    acuracias_teste.append(classificador.classificador.score(classificador.previsores_teste, classificador.classe_teste))

plt.plot(range_depth, acuracias_treinamento, label="Acuracia de Treinamento")
plt.plot(range_depth, acuracias_teste, label="Acuracia de Teste")
plt.ylabel("Acuracia")
plt.xlabel("Max_Depth")
plt.legend()

copia = acuracias_teste.copy()
copia.sort()
melhor_depth = range_depth[acuracias_teste.index(copia[-1])]
print(f"Melhor Depth: {melhor_depth} com acuracia de {copia[-1]}")

# %%
melhor_depth = 3
classificador.ArvoreDecisao(criterion=criterion, max_depth=melhor_depth, random_state=configs["random_state"])
metrificador = Metrificador(classificador)
print(round(metrificador.acuracia(), 6))
print(metrificador.matrizConfusao())

# %%

validacao_cruzada = ValidacaoCruzada(classificador, pre_processador, n_splits=5)
matriz_media, matriz_desvio_padrao, acuracia_final_media, acuracia_final_desvio_padrao, metricas_medias, metricas_desvio_padrao = validacao_cruzada.metricas()

print(matriz_media)
print(matriz_desvio_padrao)
print(acuracia_final_media)
print(acuracia_final_desvio_padrao)
print(metricas_medias)
print(metricas_desvio_padrao)
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
configs = {
    "nome_arquivo": csv_path,
    # "cols_dummy": ['City', 'Gender', 'EmploymentType', 'Education'],
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

# %%
from models.classificador import Classificador
from models.pre_processador import PreProcessador
from models.metrificador import Metrificador
from models.validacao_cruzada import ValidacaoCruzada
import matplotlib.pyplot as plt

configs["random_state"] = 0
pre_processador = PreProcessador(configs)
classificador = Classificador(pre_processador)

acuracias = []
range_k = range(1, 100)
for k in range_k:
    classificador.KNN(k)
    metrificador = Metrificador(classificador)
    acuracias.append(metrificador.acuracia())

plt.plot(range_k, acuracias)
plt.ylabel("Acuracia")
plt.xlabel("Valor de K")
plt.legend()

# %%
copia = acuracias.copy()
copia.sort()
melhor_k = range_k[acuracias.index(copia[-1])]
print(f"Melhor K: {melhor_k} com acuracia de {copia[-1]}")

# %%
classificador.KNN(melhor_k)
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
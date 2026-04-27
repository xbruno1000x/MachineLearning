# %%
from pathlib import Path
import pandas as pd
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

csv_path = Path('../data/loan_risk_balanceado.csv')
df = pd.read_csv(csv_path)
df.head(5)

# EmploymentType -> Self-Employed, Salaried, Unemployed
# Education -> PhD, Masters, Bachelors, High School
# City -> Chicago, San Francisco, New York, Houston

# %%
configs = {
    "nome_arquivo": Path('../data/loan_risk_balanceado.csv'),
    "cols_dummy": ['City', 'Gender', 'Education', 'EmploymentType'],
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
            "nome": "City"
        }
    ]
}

# %%
from models.classificador import Classificador
from models.pre_processador import PreProcessador
from models.metrificador import Metrificador
import matplotlib.pyplot as plt

configs["random_state"] = 0
pre_processador = PreProcessador(configs)
classificador = Classificador(pre_processador)

acuracias_teste = []
acuracias_treinamento = []
range_k = range(1, 100)
for k in range_k:
    classificador.KNN(k)
    metrificador = Metrificador(classificador)
    acuracia_teste = classificador.classificador.score(classificador.previsores_teste, classificador.classe_teste)
    acuracia_treinamento = classificador.classificador.score(classificador.previsores_treinamento, classificador.classe_treinamento)
    acuracias_treinamento.append(acuracia_treinamento)
    acuracias_teste.append(acuracia_teste)

plt.plot(range_k, acuracias_treinamento, label="acuracia de treinamento")
plt.plot(range_k, acuracias_teste, label="acuracia de teste")
plt.ylabel("Acuracia")
plt.xlabel("Valor de K")
plt.legend()

# %%

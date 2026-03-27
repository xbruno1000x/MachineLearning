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

# %%
df.value_counts('EmploymentType')

# %%
categoricasOrdinais = []

# %%
configs = {
    "nome_arquivo": Path('../data/loan_risk_balanceado.csv'),
    "cols_dummy": ['City', 'Gender', 'Education', 'EmploymentType'],
    "cols_categoria_ordinal": categoricasOrdinais
}

# %%
from models.classificador import Classificador
from models.pre_processador import PreProcessador
from models.metrificador import Metrificador

configs["random_state"] = 42
pre_processador = PreProcessador(configs)
classificador = Classificador(pre_processador)

resultados = []
for k in range(1, 100):
    classificador.KNN(k)
    metrificador = Metrificador(classificador)
    acuracia = metrificador.acuracia()
    matriz_confusao = metrificador.matrizConfusao()
    resultados.append({
        "k": k,
        "acuracia": acuracia,
        "matriz_confusao": matriz_confusao
    })

resultados_df = pd.DataFrame(resultados)

# %%
resultados_df.sort_values(by='acuracia', ascending=False).head(10)

# %%
print(f"Acuracia: {acuracia}"
      f"\nMatriz de Confusao:\n{matriz_confusao}")

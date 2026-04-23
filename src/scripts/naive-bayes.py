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
df.value_counts('LoanApproved')

# %%
categoricasOrdinais = []

# %%
configs = {
	"nome_arquivo": Path('../data/loan_risk_balanceado.csv'),
	"cols_dummy": ['City', 'Gender', 'Education', 'EmploymentType'],
	"cols_categoria_ordinal": categoricasOrdinais,
	"random_state": 0,
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

# %%
print(f"Acuracia: {acuracia}")
print(f"Matriz de Confusao:\n{matriz_confusao}")

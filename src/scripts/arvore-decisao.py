# %%
from pathlib import Path
import pandas as pd
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

csv_path = SRC_DIR / 'data' / 'loan_risk_balanceado.csv'
df = pd.read_csv(csv_path)
df.head(5)

# %%
df.value_counts('LoanApproved')

# %%
categoricasOrdinais = []

# %%
configs = {
	"nome_arquivo": csv_path,
	"cols_dummy": ['City', 'Gender', 'Education', 'EmploymentType'],
	"cols_categoria_ordinal": categoricasOrdinais,
	"random_state": 42,
}

# %%
from models.classificador import Classificador
from models.pre_processador import PreProcessador
from models.metrificador import Metrificador

pre_processador = PreProcessador(configs)
classificador = Classificador(pre_processador)

classificador.ArvoreDecisao(
	criterion='entropy',
	max_depth=8,
	random_state=42,
)

metrificador = Metrificador(classificador)
acuracia = metrificador.acuracia()
matriz_confusao = metrificador.matrizConfusao()

# %%
print(f"Acuracia: {acuracia}")
print(f"Matriz de Confusao:\n{matriz_confusao}")

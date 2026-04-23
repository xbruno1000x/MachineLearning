# %%
import pandas as pd
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

csv_path = Path('../data/loan_risk_balanceado.csv')
df = pd.read_csv(csv_path)

# %%
df.value_counts('LoanApproved')

# %%
df.describe()

# %%
configs = {
	"nome_arquivo": Path('../data/loan_risk_balanceado.csv'),
	"col_classe": "LoanApproved",
	"random_state": 0,
}

# %%
from models.pre_processador import PreProcessador

p = PreProcessador(configs)

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
	"random_state": 0,
}

# %%
from models.classificador import Classificador
from models.pre_processador import PreProcessador
from models.metrificador import Metrificador

pre_processador = PreProcessador(configs)
classificador = Classificador(pre_processador)

classificador.RandomForest(
	criterion='entropy',
	max_depth=10,
	n_estimators=200,
	max_features='sqrt',
	random_state=0,
)

metrificador = Metrificador(classificador)
acuracia = metrificador.acuracia()
matriz_confusao = metrificador.matrizConfusao()

importancias = pd.DataFrame(
	{
		"feature": classificador.cols_previsores,
		"importancia": classificador.classificador.feature_importances_,
	}
).sort_values(by='importancia', ascending=False)

cols_dummy = configs["cols_dummy"]

def obter_feature_original(nome_feature):
	for col_dummy in cols_dummy:
		prefixo = f"{col_dummy}_"
		if nome_feature.startswith(prefixo):
			return col_dummy
	return nome_feature

importancias_agregadas = (
	importancias
	.assign(feature_original=lambda df_: df_["feature"].apply(obter_feature_original))
	.groupby("feature_original", as_index=False)["importancia"]
	.sum()
	.sort_values(by="importancia", ascending=False)
)

# %%
print(f"Acuracia: {acuracia}")
print(f"Matriz de Confusao:\n{matriz_confusao}")
print("\nTop 10 features mais importantes:")
print(importancias.head(10).to_string(index=False))
print("\nImportancia agregada por variavel original:")
print(importancias_agregadas.to_string(index=False))

# %%

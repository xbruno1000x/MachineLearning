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
# criterion = "gini" 
criterion = "log_loss" 
# criterion = "log_loss" 
configs = {
	"max_features": "log2", #log2
    "nome_arquivo": csv_path,
    # "remover_colunas": ["Education", "YearsExperience", "Gender", "City", "Age"],
    # "cols_dummy": ['City', 'Gender', 'EmploymentType', 'Education'],
    "cols_dummy": ["City"],
    # "cols_dummy": ["EmploymentType"],
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
        # {
        #     "nome": "City"
        # },
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
range_depth = range(1, 100)

for max_depth in range_depth:
	classificador.RandomForest(
		criterion=criterion,
		max_depth=max_depth,
		n_estimators=100,
		max_features=configs["max_features"],
		random_state=configs["random_state"],
        montar_grafico=False
	)
	metrificador = Metrificador(classificador)
	acuracias_treinamento.append(classificador.classificador.score(classificador.previsores_treinamento, classificador.classe_treinamento))
	acuracias_teste.append(metrificador.acuracia())
# classificador.RandomForest(
# 	criterion='gini',
# 	max_depth=3,
# 	n_estimators=200,
# 	max_features='sqrt',
# 	random_state=0,
# )


plt.plot(range_depth, acuracias_treinamento, label="Acuracia de Treinamento")
plt.plot(range_depth, acuracias_teste, label="Acuracia de Teste")
plt.ylabel("Acuracia")
plt.xlabel("Max_Depth")
plt.legend()

# %%

classificador.RandomForest(
	criterion=criterion,
	max_depth=8,
	n_estimators=100,
	max_features="sqrt",
	random_state=0,
)
metrificador = Metrificador(classificador)
print(round(metrificador.acuracia(), 6))
print(metrificador.matrizConfusao())
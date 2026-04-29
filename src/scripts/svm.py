# %%

from models.classificador import Classificador
from models.pre_processador import PreProcessador
from models.metrificador import Metrificador
from models.validacao_cruzada import ValidacaoCruzada

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
# ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
kernel = 'rbf'
# rbf foi o melhor
# ['scale', 'auto']
gamma = 'auto'
random_state = 100

configs = {
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

pre_processador = PreProcessador(configs)
classificador = Classificador(pre_processador)
range_c = range(1, 100)
acuracias_treinamento = []
acuracias_teste = []
for c in range_c:
    classificador.SVM(
        kernel=kernel, 
        gamma=gamma, 
        C=c/100, 
        random_state=random_state
    )
    metrificador = Metrificador(classificador)
    acuracias_treinamento.append(classificador.classificador.score(classificador.previsores_treinamento, classificador.classe_treinamento))
    acuracias_teste.append(metrificador.acuracia())

from matplotlib import pyplot as plt

plt.plot(range_c, acuracias_treinamento, label="Acuracia de Treinamento")
plt.plot(range_c, acuracias_teste, label="Acuracia de Teste")
plt.ylabel("Acuracia")
plt.xlabel("Valor de C")
plt.legend()
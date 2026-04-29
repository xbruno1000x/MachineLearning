# %%
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.classificador import Classificador
from models.pre_processador import PreProcessador
from models.metrificador import Metrificador
from models.validacao_cruzada import ValidacaoCruzada

csv_path = SRC_DIR / "data" / "loan_risk_balanceado.csv"
# csv_path = SRC_DIR / "data" / "loan_risk_prediction_dataset.csv"

# EmploymentType -> Self-Employed, Salaried, Unemployed
# Education -> PhD, Masters, Bachelors, High School
# City -> Chicago, San Francisco, New York, Houston

# %%
# ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
kernel = "rbf"
# ['scale', 'auto']
gamma = "auto"
C = 1.0
random_state = 0

# Ative apenas um bloco de configs por vez.

# configs = {
#     "nome_arquivo": csv_path,
#     "cols_dummy": ["City", "Gender", "Education", "EmploymentType"],
#     "cols_categoria_ordinal": [],
#     "padronizacao": True,
#     "random_state": 0,
# }

# configs = {
#     "nome_arquivo": csv_path,
#     "cols_dummy": [],
#     "cols_categoria_ordinal": [
#         {
#             "nome": "EmploymentType",
#             "ordem": ["Self-Employed", "Salaried", "Unemployed"],
#         },
#         {
#             "nome": "Education",
#             "ordem": ["PhD", "Masters", "Bachelors", "High School"],
#         },
#         {
#             "nome": "Gender",
#         },
#         {
#             "nome": "City",
#         },
#     ],
#     "padronizacao": True,
#     "random_state": 0,
# }

# configs = {
#     "nome_arquivo": csv_path,
#     "cols_dummy": ["City"],
#     "cols_categoria_ordinal": [
#         {
#             "nome": "EmploymentType",
#             "ordem": ["Self-Employed", "Salaried", "Unemployed"],
#         },
#         {
#             "nome": "Education",
#             "ordem": ["PhD", "Masters", "Bachelors", "High School"],
#         },
#         {
#             "nome": "Gender",
#         },
#     ],
#     "padronizacao": True,
#     "random_state": 0,
# }

# configs = {
#     "nome_arquivo": csv_path,
#     "remover_colunas": ["Education", "YearsExperience", "Gender", "City"],
#     "cols_dummy": ["EmploymentType"],
#     "cols_categoria_ordinal": [],
#     "padronizacao": False,
#     "random_state": 0,
# }

configs = {
    "nome_arquivo": csv_path,
    "remover_colunas": ["Education", "YearsExperience"],
    "cols_dummy": ["EmploymentType"],
    "cols_categoria_ordinal": [
        {
            "nome": "Gender",
        },
        {
            "nome": "City",
        },
    ],
    "padronizacao": True,
    "random_state": 0,
}

processador = PreProcessador(configs)


classificador = Classificador(processador)
classificador.SVM(kernel=kernel, gamma=gamma, C=C, random_state=random_state)

metrificador = Metrificador(classificador)
acuracia = metrificador.acuracia()
matriz_confusao = metrificador.matrizConfusao()

print(f"Acuracia: {acuracia}")
print(f"Matriz de Confusao:\n{matriz_confusao}")

# %%
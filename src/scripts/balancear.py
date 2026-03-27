"""Gera uma base balanceada a partir do dataset de risco de emprestimo."""

from pathlib import Path
import sys

import pandas as pd

# Permite importar o pre-processador a partir da pasta src/models.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from models.pre_processador import PreProcessador


def balancear_por_subamostragem(base: pd.DataFrame, col_classe: str, random_state: int = 42) -> pd.DataFrame:
	"""Balanceia a base por subamostragem da classe majoritaria."""
	contagem_classes = base[col_classe].value_counts()

	if len(contagem_classes) < 2:
		raise ValueError("A base possui apenas uma classe e nao pode ser balanceada.")

	qtd_minima = int(contagem_classes.min())
	partes_balanceadas = []

	for valor_classe in contagem_classes.index:
		amostra_classe = base.loc[base[col_classe] == valor_classe]
		partes_balanceadas.append(
			amostra_classe.sample(n=qtd_minima, random_state=random_state)
		)

	base_balanceada = pd.concat(partes_balanceadas, axis=0)
	base_balanceada = base_balanceada.sample(frac=1, random_state=random_state).reset_index(drop=True)
	return base_balanceada


def main() -> None:
	base_dir = Path(__file__).resolve().parents[1]
	entrada = base_dir / "data" / "loan_risk_prediction_dataset.csv"
	saida = base_dir / "data" / "loan_risk_balanceado.csv"
	coluna_classe = "LoanApproved"

	configs = {
		"nome_arquivo": str(entrada),
		"col_classe": coluna_classe,
		"random_state": 42,
	}

	processador = PreProcessador(configs)
	base_tratada = processador.base.copy()

	base_balanceada = balancear_por_subamostragem(
		base=base_tratada,
		col_classe=coluna_classe,
		random_state=42,
	)

	base_balanceada.to_csv(saida, index=False)

	print(f"Arquivo balanceado salvo em: {saida}")
	print("Distribuicao original:", base_tratada[coluna_classe].value_counts().to_dict())
	print("Distribuicao balanceada:", base_balanceada[coluna_classe].value_counts().to_dict())


if __name__ == "__main__":
	main()

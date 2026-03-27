base_caracteristicas_e_objetivo = pd.concat([previsores, classe], axis=1)
correlacao = base_caracteristicas_e_objetivo.corr()

import seaborn as sns

ax = sns.heatmap(
    correlacao,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    xticklabels=True,
    yticklabels=True
)
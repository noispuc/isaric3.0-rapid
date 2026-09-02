import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class SHAPPlots:
    """
    Visualizações de interpretabilidade baseadas em SHAP (SHapley Additive
    exPlanations), usadas pelos modelos preditivos baseados em árvore
    (Random Forest, XGBoost).

    Diferente dos demais plots do pacote (interativos, via Plotly), os
    plots SHAP usam a implementação nativa da biblioteca `shap` (matplotlib)
    e são salvos como artefatos PNG em disco, conforme exigido pelo
    workflow de interpretabilidade do RAPID.

    Privacidade (NFR008 / RAPID Methodology §2.2): a saída de um RAPID deve
    ser agregada, sem informação a nível de paciente. Por isso:

    - `summary_plot` usa a importância média absoluta por feature — agregado.
    - `aggregate_shap` + `aggregated_beeswarm_plot` substituem o beeswarm
      clássico por faixas por bin de valor da feature, com supressão de bins
      pequenos — agregado.
    - `beeswarm_plot` (um ponto por paciente) **não** é agregado e não deve
      compor artefatos de `report()`/`save()`. Ver aviso no próprio método.
    """

    @staticmethod
    def summary_plot(shap_values, features, feature_names=None,
                      title: str = "SHAP Summary Plot",
                      output_path: str = "shap_summary_plot.png",
                      max_display: int = 20) -> str:
        """
        Gera e salva um summary plot (importância média absoluta de cada
        feature) a partir de valores SHAP.

        Args:
            shap_values (numpy.ndarray): Valores SHAP (n_amostras, n_features),
                já filtrados para a classe positiva quando aplicável.
            features (numpy.ndarray or pandas.DataFrame): Matriz de features
                usada para calcular os valores SHAP.
            feature_names (list, optional): Nomes das features.
            title (str): Título do plot.
            output_path (str): Caminho de saída do artefato PNG.
            max_display (int): Número máximo de features exibidas.

        Returns:
            str: Caminho do arquivo PNG salvo.
        """
        import shap

        plt.figure()
        shap.summary_plot(
            shap_values, features, feature_names=feature_names,
            plot_type="bar", max_display=max_display, show=False
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path

    @staticmethod
    def aggregate_shap(shap_values, features, feature_names=None,
                       max_display: int = 20, n_bins: int = 10,
                       min_bin_size: int = 10) -> pd.DataFrame:
        """
        Agrega valores SHAP por bin de valor da feature, produzindo a tabela
        que substitui o beeswarm individual.

        Para cada feature, os registros são agrupados por faixa de valor
        (quantis, ou o próprio valor quando a feature é discreta) e o efeito
        SHAP é resumido por mediana e intervalo interquartil. Bins com menos
        de `min_bin_size` registros são **descartados**: sem esse controle,
        um "decil" com poucos pacientes continuaria sendo informação a nível
        individual, e a agregação seria apenas nominal.

        Args:
            shap_values (numpy.ndarray): Valores SHAP (n_amostras, n_features),
                já filtrados para a classe positiva quando aplicável.
            features (numpy.ndarray or pandas.DataFrame): Matriz de features
                usada para calcular os valores SHAP.
            feature_names (list, optional): Nomes das features.
            max_display (int): Número máximo de features retornadas, por
                ordem de importância média absoluta.
            n_bins (int): Número de faixas de valor por feature contínua.
            min_bin_size (int): Mínimo de registros para um bin ser reportado.

        Returns:
            pandas.DataFrame: Uma linha por (feature, bin), com colunas
            `feature`, `bin_index`, `bin_label`, `n`, `feature_value_min`,
            `feature_value_max`, `shap_median`, `shap_p25`, `shap_p75` e
            `mean_abs_shap`. Nenhuma linha corresponde a um paciente.
        """
        X = np.asarray(features.values if hasattr(features, "values") else features, dtype=float)
        S = np.asarray(shap_values, dtype=float)

        if X.shape != S.shape:
            raise ValueError(
                f"features {X.shape} e shap_values {S.shape} precisam ter a mesma forma."
            )

        if feature_names is None:
            feature_names = [f"feature_{j}" for j in range(X.shape[1])]
        feature_names = list(feature_names)

        mean_abs = np.nanmean(np.abs(S), axis=0)
        order = np.argsort(mean_abs)[::-1][:max_display]

        rows = []
        for j in order:
            col, shap_col = X[:, j], S[:, j]
            valid = ~np.isnan(col)
            col, shap_col = col[valid], shap_col[valid]
            if col.size == 0:
                continue

            uniques = np.unique(col)
            if uniques.size <= n_bins:
                # Feature discreta (binárias de sintoma/comorbidade caem aqui):
                # agrupar pelo próprio valor é mais informativo que quantis.
                bin_ids = np.searchsorted(uniques, col)
                labels = [f"{v:g}" for v in uniques]
            else:
                bin_ids = pd.qcut(col, n_bins, labels=False, duplicates="drop")
                bin_ids = np.asarray(bin_ids)
                edges = pd.qcut(col, n_bins, duplicates="drop").categories
                labels = [f"({c.left:.3g}, {c.right:.3g}]" for c in edges]

            for b in range(len(labels)):
                mask = bin_ids == b
                n = int(mask.sum())
                if n < min_bin_size:
                    continue  # supressão por tamanho de bin
                rows.append({
                    "feature": feature_names[j],
                    "bin_index": b,
                    "bin_label": labels[b],
                    "n": n,
                    "feature_value_min": float(col[mask].min()),
                    "feature_value_max": float(col[mask].max()),
                    "shap_median": float(np.median(shap_col[mask])),
                    "shap_p25": float(np.percentile(shap_col[mask], 25)),
                    "shap_p75": float(np.percentile(shap_col[mask], 75)),
                    "mean_abs_shap": float(mean_abs[j]),
                })

        return pd.DataFrame(rows)

    @staticmethod
    def aggregated_beeswarm_plot(aggregate_df: pd.DataFrame,
                                  title: str = "SHAP Aggregated Beeswarm",
                                  output_path: str = "shap_aggregated_beeswarm.png") -> str:
        """
        Gera o beeswarm agregado a partir da tabela de `aggregate_shap`.

        Preserva o que o beeswarm clássico comunica — direção e dispersão do
        efeito de cada feature, e como variam com o valor da feature — sem
        desenhar um ponto por paciente: cada faixa de valor vira um segmento
        interquartil com marcador na mediana, colorido do menor ao maior
        valor da feature.

        Args:
            aggregate_df (pandas.DataFrame): Saída de `aggregate_shap`.
            title (str): Título do plot.
            output_path (str): Caminho de saída do artefato PNG.

        Returns:
            str: Caminho do arquivo PNG salvo.
        """
        if aggregate_df.empty:
            raise ValueError(
                "Nenhum bin sobreviveu à agregação. Reduza n_bins ou min_bin_size, "
                "ou use uma amostra maior."
            )

        features = (
            aggregate_df.groupby("feature")["mean_abs_shap"].first()
            .sort_values(ascending=True).index.tolist()
        )
        cmap = matplotlib.colormaps["coolwarm"]

        fig, ax = plt.subplots(figsize=(9, max(3.0, 0.55 * len(features) + 1.6)))
        for y, feature in enumerate(features):
            sub = aggregate_df[aggregate_df["feature"] == feature].sort_values("bin_index")
            n_bins = len(sub)
            for k, (_, row) in enumerate(sub.iterrows()):
                # Espalha os bins verticalmente dentro da linha da feature,
                # do menor valor (embaixo) ao maior (em cima).
                offset = 0.0 if n_bins == 1 else (k / (n_bins - 1) - 0.5) * 0.62
                color = cmap(0.5 if n_bins == 1 else k / (n_bins - 1))
                ax.hlines(y + offset, row["shap_p25"], row["shap_p75"],
                          color=color, linewidth=2.6, alpha=0.85)
                ax.plot(row["shap_median"], y + offset, "o", color=color,
                        markersize=4.5, markeredgecolor="white", markeredgewidth=0.5)

        ax.axvline(0, color="0.4", linewidth=0.9, linestyle="--", zorder=0)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel("Valor SHAP (impacto na predição)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)

        colorbar = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap=cmap),
            ax=ax, pad=0.02, fraction=0.03,
        )
        colorbar.set_ticks([0, 1])
        colorbar.set_ticklabels(["menor", "maior"])
        colorbar.set_label("Valor da feature")

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    @staticmethod
    def beeswarm_plot(shap_values, features, feature_names=None,
                       title: str = "SHAP Beeswarm Plot",
                       output_path: str = "shap_beeswarm_plot.png",
                       max_display: int = 20) -> str:
        """
        Beeswarm clássico: um ponto por amostra, mostrando direção e magnitude
        do impacto de cada feature.

        AVISO — saída a nível de paciente. Este plot desenha um marcador por
        registro, com a cor codificando o valor real da feature daquele
        registro. Não é saída agregada e, portanto, **não deve compor
        artefatos de `report()` nem de `save()`** (NFR008 e RAPID Methodology
        §2.2). Para relatórios, use `aggregate_shap` +
        `aggregated_beeswarm_plot`. Mantido apenas para inspeção local, pelo
        próprio pesquisador, sobre os próprios dados.

        Args:
            shap_values (numpy.ndarray): Valores SHAP (n_amostras, n_features),
                já filtrados para a classe positiva quando aplicável.
            features (numpy.ndarray or pandas.DataFrame): Matriz de features
                usada para calcular os valores SHAP.
            feature_names (list, optional): Nomes das features.
            title (str): Título do plot.
            output_path (str): Caminho de saída do artefato PNG.
            max_display (int): Número máximo de features exibidas.

        Returns:
            str: Caminho do arquivo PNG salvo.
        """
        import shap

        plt.figure()
        shap.summary_plot(
            shap_values, features, feature_names=feature_names,
            plot_type="dot", max_display=max_display, show=False
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path

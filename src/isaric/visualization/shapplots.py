import matplotlib.pyplot as plt


class SHAPPlots:
    """
    Visualizações de interpretabilidade baseadas em SHAP (SHapley Additive
    exPlanations), usadas pelos modelos preditivos baseados em árvore
    (Random Forest, XGBoost).

    Diferente dos demais plots do pacote (interativos, via Plotly), os
    plots SHAP usam a implementação nativa da biblioteca `shap` (matplotlib)
    e são salvos como artefatos PNG em disco, conforme exigido pelo
    workflow de interpretabilidade do RAPID.
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
    def beeswarm_plot(shap_values, features, feature_names=None,
                       title: str = "SHAP Beeswarm Plot",
                       output_path: str = "shap_beeswarm_plot.png",
                       max_display: int = 20) -> str:
        """
        Gera e salva um beeswarm plot, mostrando a distribuição do impacto
        de cada feature nas previsões (direção e magnitude por amostra).

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

import numpy as np


def detect_collinearity(df, threshold=0.9):
    """
    Description:
        Detects highly correlated features based on a threshold.

    Args:
        df (pandas.DataFrame): Input dataset.
        threshold (float): Correlation coefficient threshold.

    Returns:
        list: List of column names to consider for removal.
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    return [column for column in upper.columns if any(upper[column] > threshold)]


def collinearity_report(X, corr_threshold=0.9, vif_threshold=5.0):
    """
    Description:
        Gera um relatório de colinearidade combinando matriz de correlação
        de Pearson e Variance Inflation Factor (VIF), apenas para
        diagnóstico — nenhuma variável é removida automaticamente, a decisão
        fica a critério do analista.

    Args:
        X (pandas.DataFrame): Matriz de preditores (idealmente já numérica;
            colunas não numéricas são ignoradas no cálculo).
        corr_threshold (float): Limiar acima do qual um par de variáveis é
            sinalizado por correlação alta.
        vif_threshold (float): Limiar acima do qual uma variável é
            sinalizada por VIF alto.

    Returns:
        dict: {
            'correlation_matrix': pandas.DataFrame com a matriz de correlação,
            'high_correlation_pairs': pandas.DataFrame (var1, var2, correlation)
                para pares acima de corr_threshold,
            'vif': pandas.DataFrame (feature, VIF, flag) via
                ModelAssumptionTester.test_vif.
        }
    """
    # Import local para evitar ciclo de import: modelevaluation também
    # depende (indiretamente) de isaric.modeling, que agora depende deste
    # módulo via predictive_classifier.py.
    from isaric.modelevaluation.assumptiontester import ModelAssumptionTester

    numeric_X = X.select_dtypes(include=[np.number]).dropna(axis=1, how="all")

    corr_matrix = numeric_X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_correlation_pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "var1", "level_1": "var2", 0: "correlation"})
    )
    high_correlation_pairs = high_correlation_pairs[
        high_correlation_pairs["correlation"] > corr_threshold
    ].sort_values("correlation", ascending=False).reset_index(drop=True)

    # VIF exige matriz sem NaN; preenchimento pela mediana é usado apenas
    # para este diagnóstico, não afeta a imputação usada na modelagem.
    tester = ModelAssumptionTester(
        model=None,
        X=numeric_X.fillna(numeric_X.median()),
        y=np.zeros(len(numeric_X)),
        y_pred=np.zeros(len(numeric_X)),
    )
    vif_df = tester.test_vif()
    vif_df["flag"] = vif_df["VIF"] > vif_threshold

    return {
        "correlation_matrix": corr_matrix,
        "high_correlation_pairs": high_correlation_pairs,
        "vif": vif_df,
    }

"""
lca_radar.py
RAP (Reproducible Analytical Pipeline) para LCA (Latent Class Analysis) em dados binários,
com gráficos radar por classe. Focado em interpretabilidade para profissionais de saúde.

- Leitura direta de Parquet
- Autodetecção de colunas binárias (0/1/NaN/True/False)
- LCA com StepMix (medição Bernoulli), nº de classes parametrizável (default=13)
- Perfis de classe calculados via posterior probabilities (robusto)
- Exporta PNGs dos radares e CSVs com perfis e pesos de classe

Requisitos:
    pip install stepmix pandas numpy matplotlib scikit-learn pyarrow fastparquet
"""

from __future__ import annotations
import os
import warnings
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stepmix.stepmix import StepMix
from isaric.visualization.lcaplots import LCAPlots

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# Utilidades de dados
# -----------------------------
def read_parquet_binary(
    parquet_path: str,
    feature_cols: Optional[Sequence[str]] = None,
    id_cols: Optional[Sequence[str]] = None,
    drop_na_rows: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Lê um Parquet e retorna (df, cols_binarias) prontos para LCA.

    - Se feature_cols for None: autodetecta colunas binárias (0/1, True/False; aceita NaN).
    - Converte bool -> 0/1.
    - id_cols (se fornecido) são preservadas, mas excluídas da seleção automática.

    Args:
        parquet_path: caminho do arquivo .parquet
        feature_cols: lista explícita de colunas a usar como binárias
        id_cols: colunas de identificação a preservar e não usar na LCA
        drop_na_rows: se True, dropa linhas com NaN nas features; senão, preenche com 0

    Returns:
        df: DataFrame completo (inclui id_cols se existirem)
        binary_cols: lista final de colunas binárias usadas na LCA
    """
    df = pd.read_parquet(parquet_path)

    if id_cols:
        id_cols = [c for c in id_cols if c in df.columns]
    else:
        id_cols = []

    if feature_cols is not None:
        # Usa exatamente o que foi passado
        binary_cols = [c for c in feature_cols if c in df.columns]
        _coerce_bools_inplace(df, binary_cols)
    else:
        # Autodetecta binárias
        candidate_cols = [c for c in df.columns if c not in id_cols]
        binary_cols = _autodetect_binary_columns(df, candidate_cols)
        _coerce_bools_inplace(df, binary_cols)

    # Lidar com NaNs nas features binárias
    if drop_na_rows:
        df = df.dropna(subset=binary_cols)
    else:
        df[binary_cols] = df[binary_cols].fillna(0)

    # Garantir tipo numérico 0/1
    df[binary_cols] = df[binary_cols].astype(float)

    return df, binary_cols


def _coerce_bools_inplace(df: pd.DataFrame, cols: Sequence[str]) -> None:
    """Converte colunas booleanas para 0/1 inplace."""
    for c in cols:
        if pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype(int)


def _is_binary_series(s: pd.Series) -> bool:
    """True se a série é binária (aceita {0,1}, {True,False}, com ou sem NaN)."""
    vals = pd.Series(s.dropna().unique())
    if vals.empty:
        return False
    # Normaliza bool -> int
    if pd.api.types.is_bool_dtype(vals):
        return True
    try:
        numeric_vals = pd.to_numeric(vals, errors="coerce")
    except Exception:
        return False
    unique_set = set(numeric_vals.dropna().astype(int).unique().tolist())
    return unique_set.issubset({0, 1}) and len(unique_set) > 0


def _autodetect_binary_columns(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    """Retorna colunas binárias detectadas automaticamente."""
    binary = [c for c in cols if _is_binary_series(df[c])]
    if len(binary) == 0:
        raise ValueError(
            "Nenhuma coluna binária detectada. "
            "Informe 'feature_cols' manualmente ou revise o dataset."
        )
    return binary


# -----------------------------
# LCA com StepMix
# -----------------------------
def fit_lca_bernoulli(
    X: Union[pd.DataFrame, np.ndarray],
    n_classes: int = 13,
    n_init: int = 5,
    max_iter: int = 500,
    random_state: int = 42,
    verbose: int = 0,
) -> StepMix:
    """
    Ajusta um modelo LCA (Bernoulli) com StepMix.

    Args:
        X: matriz (n_amostras, n_variáveis) com 0/1
        n_classes: número de classes latentes (default 13)
        n_init: quantas inicializações aleatórias
        max_iter: iterações do EM
        random_state: semente
        verbose: 0 (silencioso) a 2 (bem verboso)

    Returns:
        modelo StepMix ajustado
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    model = StepMix(
        n_components=n_classes,
        measurement="bernoulli",   # variáveis observadas binárias
        structural=None,           # sem regressão/estrutura adicional
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        verbose=verbose,
    )
    model.fit(X)
    return model


def class_profiles_from_posteriors(
    X: Union[pd.DataFrame, np.ndarray],
    posteriors: np.ndarray,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """
    Calcula, para cada classe, a probabilidade de '1' em cada variável (perfil de classe),
    usando médias ponderadas pelas probabilidades posteriores q(z=c | x).

    Args:
        X: matriz binária (n_amostras, n_variáveis)
        posteriors: matriz (n_amostras, n_classes)
        feature_names: nomes das variáveis (len = n_variáveis)

    Returns:
        DataFrame de shape (n_classes, n_variáveis) com probabilidades em [0,1]
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    n_samples, n_features = X.shape
    n_classes = posteriors.shape[1]
    profiles = np.zeros((n_classes, n_features), dtype=float)

    for c in range(n_classes):
        w = posteriors[:, c].reshape(-1, 1)  # (n,1)
        denom = w.sum()
        # evita divisão por zero
        if denom <= 0:
            profiles[c, :] = np.nan
        else:
            profiles[c, :] = (w * X).sum(axis=0) / denom

    prof_df = pd.DataFrame(profiles, columns=feature_names)
    prof_df.index = [f"Classe_{i+1}" for i in range(n_classes)]
    return prof_df


# -----------------------------
# Radar plots
# -----------------------------
def plot_radar_per_class(
    class_profiles: pd.DataFrame,
    order_features_by: str = "input",  # "input" | "variance" | "mean"
    max_vars: Optional[int] = None,
    output_dir: str = "lca_outputs",
    basename: str = "radar_class",
    save_formats: Tuple[str, ...] = ("png",),
    dpi: int = 150,
) -> List[str]:
    """
    Gera um gráfico radar para cada classe e salva em arquivo.

    Args:
        class_profiles: DF (n_classes x n_features) com probabilidades em [0,1]
        order_features_by: ordenação dos eixos: "input" (sem mudar),
                           "variance" (maior var primeiro) ou "mean" (maior média primeiro)
        max_vars: se quiser limitar o número de variáveis mais informativas
        output_dir: pasta de saída
        basename: prefixo dos arquivos
        save_formats: formatos de imagem, ex. ("png", "pdf")
        dpi: resolução

    Returns:
        Lista com caminhos dos arquivos gerados
    """
    os.makedirs(output_dir, exist_ok=True)

    prof = class_profiles.copy()

    # Ordenação dos eixos (mesmo para todas as classes, para facilitar leitura)
    if order_features_by == "variance":
        order = prof.var(axis=0).sort_values(ascending=False).index.tolist()
    elif order_features_by == "mean":
        order = prof.mean(axis=0).sort_values(ascending=False).index.tolist()
    else:
        order = list(prof.columns)

    if max_vars is not None and max_vars > 0:
        order = order[:max_vars]

    prof = prof[order]

    # Preparar ângulos do radar
    labels = prof.columns.tolist()
    n_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]  # fecha o ciclo

    out_files: List[str] = []

    for idx, (cls, row) in enumerate(prof.iterrows(), start=1):
        values = row.values.tolist()
        # fecha o ciclo para plotagem
        values += values[:1]

        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Eixos e rótulos
        plt.xticks(angles[:-1], labels, fontsize=9)
        ax.set_rlabel_position(0)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(["0.25", "0.50", "0.75"], fontsize=8)
        ax.set_ylim(0, 1)

        # Linha e preenchimento
        ax.plot(angles, values, linewidth=2)
        ax.fill(angles, values, alpha=0.2)

        plt.title(f"{cls} — Probabilidade de ocorrência", y=1.08, fontsize=12)

        # Salvar
        for fmt in save_formats:
            out_path = os.path.join(output_dir, f"{basename}_{idx:02d}.{fmt}")
            plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
            out_files.append(out_path)

        plt.close(fig)

    return out_files


# -----------------------------
# Pipeline completo (RAP-friendly)
# -----------------------------
def run_lca_pipeline(
    parquet_path: str = "data_SinanDengue_2019_treated_part.parquet",
    n_classes: int = 13,
    feature_cols: Optional[Sequence[str]] = None,
    id_cols: Optional[Sequence[str]] = None,
    drop_na_rows: bool = True,
    n_init: int = 5,
    max_iter: int = 500,
    random_state: int = 42,
    verbose: int = 0,
    radar_order_by: str = "variance",   # "input" | "variance" | "mean"
    radar_max_vars: Optional[int] = None,
    output_dir: str = "lca_outputs",
) -> dict:
    """
    Executa o fluxo completo:
     1) Lê Parquet
     2) Seleciona colunas binárias
     3) Ajusta LCA (StepMix)
     4) Calcula perfis de classe (probabilidade de '1' por variável)
     5) Gera gráficos radar por classe + exports CSV

    Returns (dict):
        {
          "model": StepMix,
          "class_profiles": DataFrame (classes x variáveis),
          "class_weights": Series (peso de cada classe),
          "assignments": Series (classe argmax por indivíduo),
          "radar_files": [lista de paths],
          "features_used": [lista de colunas]
        }
    """
    # 1) Leitura + features binárias
    df, binary_cols = read_parquet_binary(
        parquet_path=parquet_path,
        feature_cols=feature_cols,
        id_cols=id_cols,
        drop_na_rows=drop_na_rows,
    )

    X = df[binary_cols].copy()

    # 2) Ajuste do modelo
    model = fit_lca_bernoulli(
        X=X,
        n_classes=n_classes,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        verbose=verbose,
    )

    # 3) Posteriores, pesos e atribuições
    post = model.predict_proba(X.values)  # (n, k)
    class_weights = pd.Series(post.mean(axis=0), index=[f"Classe_{i+1}" for i in range(n_classes)], name="peso")
    assignments = pd.Series(post.argmax(axis=1) + 1, name="classe_predita")  # 1..k

    # 4) Perfis de classe
    class_profiles = class_profiles_from_posteriors(
        X=X,
        posteriors=post,
        feature_names=binary_cols,
    )

    # 5) Exports
    os.makedirs(output_dir, exist_ok=True)
    class_profiles.to_csv(os.path.join(output_dir, "class_profiles.csv"), index=True, encoding="utf-8-sig")
    class_weights.to_csv(os.path.join(output_dir, "class_weights.csv"), header=True, encoding="utf-8-sig")

    # 6) Plots radar (Plotly interativo)
    radar_fig = LCAPlots.plot_radar_profiles(class_profiles)
    radar_html = os.path.join(output_dir, "radar_profiles.html")
    radar_fig.write_html(radar_html)
    radar_files = [radar_html]

    # Também pode exportar as atribuições indivíduo-a-classe, se houver id_cols
    if id_cols:
        out_assign = df[id_cols].copy()
        out_assign["classe_predita"] = assignments.values
        out_assign.to_csv(os.path.join(output_dir, "individual_assignments.csv"), index=False, encoding="utf-8-sig")

    return {
        "model": model,
        "class_profiles": class_profiles,
        "class_weights": class_weights,
        "assignments": assignments,
        "radar_files": radar_files,
        "features_used": binary_cols,
    }


# -----------------------------
# Execução direta (exemplo)
# -----------------------------
if __name__ == "__main__":
    """
    Rode no terminal integrado do VS Code:
        python lca_radar.py
    Os resultados (CSVs e PNGs) ficarão na pasta ./lca_outputs
    """
    results = run_lca_pipeline(
        parquet_path="data_SinanDengue_2019_treated_part.parquet",  # conforme solicitado
        n_classes=13,               # fixo por padrão (ajuste se desejar)
        feature_cols=None,          # ou lista explícita de colunas binárias
        id_cols=None,               # ex.: ["id_paciente", "ano"]
        drop_na_rows=True,
        n_init=8,
        max_iter=600,
        random_state=42,
        verbose=0,
        radar_order_by="variance",  # "input" | "variance" | "mean"
        radar_max_vars=30,          # limite de variáveis por radar (opcional)
        output_dir="lca_outputs",
    )
    print("Concluído. Arquivos salvos em:", os.path.abspath("lca_outputs"))
    print("Classes e pesos:\n", results["class_weights"])


# ===========================================================
# FACADE CLASS: RAPID_PhenotypeLCA (camada de alto nível)
# ===========================================================
from stepmix.stepmix import StepMix

from isaric.visualization.lcaplots import LCAPlots
from isaric.modeling.base_model import RAPID_BasePipeline


class RAPID_PhenotypeLCA(RAPID_BasePipeline): # Inherits from RAPID_BasePipeline if available
    """
    Pipeline for Phenotype Clustering using Latent Class Analysis (LCA).
    This class identifies hidden subgroups (phenotypes) within a population 
    based on observed categorical variables.
    """

    def __init__(self, data: pd.DataFrame, measurement_vars: list, structural_var: str = None, n_components: int = None):
        """
        Initialize the LCA pipeline.
        
        :param data: Input DataFrame
        :param measurement_vars: List of binary/categorical variables for the measurement model
        :param structural_var: Optional variable for the structural model (e.g., outcome like HOSPITALIZ)
        :param n_components: Number of latent classes to identify (defaults to None to trigger grid search in fit)
        """
        self.data = data.copy()
        self.measurement_vars = measurement_vars
        self.structural_var = structural_var
        self.n_components = n_components
        self.is_decided = False
        
        self.fitted_model = None
        self.clusters = None
        self.fit_metrics = None
        self.grid_results = None
        self.modelObjs = {}

    # ------------------------------------------------------------------
    # PUBLIC METHODS
    # -----------------------------------------------------------------
    def fit(self, cluster_range: range = None):
        """
        Public method to run the preprocessing and model fitting.
        If cluster_range is provided, runs a grid search.
        If cluster_range is not provided, defaults to a grid search 
        from 2 to 6 classes, unless n_components is set on init.
        """
        self._preprocess_data()
        
        if cluster_range is not None:
            self.grid_search(cluster_range)
        elif self.n_components is not None:
            self._modeling()
            self._model_evaluation()
        else:
            default_range = range(2, 6)
            print(f"No cluster_range provided. Running default grid search for classes {list(default_range)}...")
            self.grid_search(default_range)

        return self

    def summary(self, k: int = None):
        """
        Public method to render the model summary visualizations.
        If k is informed, shows results of the specific selected k.
        Otherwise, shows grid search metrics if available.
        """
        if k is not None:
            self.describe(k)
        elif getattr(self, 'grid_results', None) is not None:
            print("Showing grid search metrics. Use summary(k=...) to view specific class results.")
            self.summary_grid_plots()
        elif getattr(self, 'fitted_model', None) is not None:
            self.describe(self.n_components)
        else:
            raise ValueError("Model not fitted. Call .fit() first.")
    # ------------------------------------------------------------------
    # PRIVATE METHODS (FOLLOWING THE STANDARD ISARIC PIPELINE STRUCTURE)
    # ------------------------------------------------------------------
    
    def _preprocess_data(self):
        self._data_cleaning()
        self._preprocessing()

    def _data_cleaning(self):
        """
        Logic from Notebook 2: Remove rows with missing values in key variables.
        """
        relevant_cols = self.measurement_vars + ([self.structural_var] if self.structural_var else [])
        initial_len = len(self.data)
        self.data.dropna(subset=relevant_cols, inplace=True)
        print(f"Data Cleaning: Removed {initial_len - len(self.data)} rows with NaNs.")
    
    def _preprocessing(self):
        """
        Internal: Prepares data by removing rows with missing values in key variables.
        """
        self.X = self.data[self.measurement_vars]
        self.y = self.data[self.structural_var] if self.structural_var else None


    # ------------------------------------------------------------------
    # PRIVATE METHODS: MODELING & EVALUATION
    # ------------------------------------------------------------------
    def _modeling(self):
        """
        Internal: Fits the StepMix model using Bernoulli distribution for binary variables.
        """
        self.fitted_model = StepMix(
            n_components=self.n_components, 
            measurement="bernoulli", 
            structural="bernoulli" if self.y is not None else None,
            random_state=42,
            verbose=0
        )

        self.fitted_model.fit(self.X, self.y)
        self._compute_assignments()

    def _model_evaluation(self):
        """
        Calculate Information Criteria for the current model.
        """
        if self.fitted_model:
            aic = self.fitted_model.aic(self.X, self.y)
            bic = self.fitted_model.bic(self.X, self.y)
            self.fit_metrics = pd.DataFrame({
                'Metric': ['AIC', 'BIC'],
                'Value': [aic, bic],
                'K': [self.n_components, self.n_components]
            })

    def _validation(self, cluster_range: range):
        """
        Required by RAPID_BasePipeline abstract class.
        Proxies to the enhanced comprehensive grid_search method.
        """
        return self.grid_search(cluster_range)

    def grid_search(self, cluster_range: range, max_iter: int = 2000):
        """
        Grid search over multiple class counts, capturing comprehensive metrics.
        """
        from sklearn.base import clone
        from isaric.modelevaluation.assumptiontester import ModelAssumptionTester
        
        if not hasattr(self, 'X') or self.X is None:
            self._preprocess_data()
            
        results = []
        self.modelObjs = {}
        
        base_model = StepMix(
            measurement="bernoulli", 
            structural="bernoulli" if self.y is not None else None,
            random_state=42,
            max_iter=max_iter,
            verbose=0,
            progress_bar=0
        )
        
        n_samples = self.X.shape[0]
        n_col = self.X.shape[1]
        
        for k in cluster_range:
            print(f"Testing {k} classes...")
            temp_model = clone(base_model)
            temp_model.set_params(n_components=k)
            temp_model.fit(self.X, self.y)
            
            avg_ll = temp_model.score(self.X, self.y)
            ll = avg_ll * n_samples
            npar = temp_model.n_parameters
            ncomp = temp_model.n_components
            
            aic = -2 * avg_ll * n_samples + 2 * npar
            bic = -2 * avg_ll * n_samples + npar * np.log(n_samples)
            caic = -2 * avg_ll * n_samples + npar * (np.log(n_samples) + 1)
            sabic = -2 * avg_ll * n_samples + npar * np.log(n_samples * ((n_samples + 2) / 24))
            
            entropy = temp_model.entropy(self.X)
            relentropy = 1 - entropy / (n_samples * np.log(ncomp)) if ncomp > 1 else np.nan
            
            # dof logic from AdjGridSearch5
            dof = (2**n_col) - ((ncomp - 1) + n_col * 2) 
            
            results.append({
                'n_clusters': k, 'LL': ll, 'score': avg_ll,
                'AIC': aic, 'BIC': bic, 'CAIC': caic, 'SABIC': sabic,
                'entropy': entropy, 'relative_entropy': relentropy,
                'convergence': temp_model.converged_,
                'npar': npar, 'n': n_samples, 'ncomp': ncomp, 'dof': dof
            })
            self.modelObjs[k] = temp_model
            
        stats = pd.DataFrame(results)
        
        # Likelihood ratio tests
        nested_lrt = []
        for k in range(min(cluster_range) + 1, max(cluster_range) + 1):
            if k in stats['ncomp'].values and (k-1) in stats['ncomp'].values:
                ll_k = float(stats.loc[stats['ncomp'] == k, 'LL'].values[0])
                ll_kp = float(stats.loc[stats['ncomp'] == k-1, 'LL'].values[0])
                dof_k = int(stats.loc[stats['ncomp'] == k, 'dof'].values[0])
                dof_kp = int(stats.loc[stats['ncomp'] == k-1, 'dof'].values[0])
                
                lrt = ModelAssumptionTester.likelihood_ratio_test(ll_kp, ll_k, dof_kp, dof_k)
                lrt['ncomp'] = k
                nested_lrt.append(lrt)
                
        if nested_lrt:
            stats = pd.merge(stats, pd.DataFrame(nested_lrt), on='ncomp', how='left')
            
        self.grid_results = stats
        return self.grid_results

    def decide(self, k: int):
        """
        Select definitive model to proceed.
        """
        if not hasattr(self, 'modelObjs') or k not in self.modelObjs:
            raise ValueError(f"Model with {k} clusters not found. Run fit() first.")
        self.n_components = k
        self.fitted_model = self.modelObjs[k]
        self._compute_assignments()
        self.is_decided = True
        print(f"Decision stored: Model selected with k={k} clusters.")
        
    def describe(self, k: int = None):
        """
        Show exploratory plots mapping to the exploratory notebooks.
        """
        target_k = k if k is not None else self.n_components
        if hasattr(self, 'modelObjs') and target_k in self.modelObjs:
            target_model = self.modelObjs[target_k]
        else:
            if self.fitted_model is None or self.n_components != target_k:
                raise ValueError("Model not fitted.")
            target_model = self.fitted_model

        print(f"======================= LCA Description (k = {target_k}) =========================")
        df_params = target_model.get_parameters_df()
        pis_df = df_params.loc[('measurement', slice(None), 'pis')]
        prob_df = pis_df.unstack('variable')['value']
        prob_df.columns = self.measurement_vars
        
        LCAPlots.plot_conditional_probs_line(prob_df).show()
        LCAPlots.plot_radar_profiles(prob_df).show()
        
        target_clusters = pd.Series(target_model.predict(self.X), index=self.data.index)
        LCAPlots.plot_clusters(target_clusters).show()
        
        if self.y is not None:
            print('------------------------ Cross-Tabulation (Outcome vs Predicted) -----------------')
            print(pd.crosstab(self.y, target_clusters, normalize='columns'))

    def summary_grid_plots(self):
        """
        Show the grid search criteria & entropy charts.
        """
        if self.grid_results is None:
            raise ValueError("Run grid_search first.")
        LCAPlots.plot_model_selection(self.grid_results).show()
        LCAPlots.plot_grid_metrics(self.grid_results).show()
        LCAPlots.plot_grid_entropy(self.grid_results).show()


    def report(self):
        """Displays all metrics and all plots without filters."""
        if self.grid_results is not None:
            self.summary_grid_plots()
        if self.fitted_model is not None:
            self.describe(self.n_components)

    def validate(self, **kwargs):
        """Placeholder for external/subgroup validation."""
        raise NotImplementedError("LCA validation not yet implemented.")

    def _visualization(self):
        """
        Orchestrate all Plotly visualizations.
        """
        self._render_profiles()
        self._render_distribution()

        if self.grid_results is not None:
            fig = LCAPlots.plot_model_selection(self.grid_results)
            fig.show()


    

    # ------------------------------------------------------------------
    # HELPER & EXPORT METHODS
    # ------------------------------------------------------------------
    def _compute_assignments(self):
        """
        Internal: Predicts classes and attaches them to the dataframe.
        """
        self.clusters = pd.Series(self.fitted_model.predict(self.X), index=self.data.index)
        self.data['latent_class'] = self.clusters

    def _render_profiles(self):
        if self.fitted_model is None: raise ValueError("Fit model first.")
        #StepMix stores measurement model parameters in 'mm_stats'
        # We extract the probabilities (usually under the 'pis' key for Bernoulli)
        df_params = self.fitted_model.get_parameters_df()
        pis_df = df_params.loc[('measurement', slice(None), 'pis')]
        prob_df = pis_df.unstack('variable')['value']
        
        # Ensure the columns match your clinical variables
        prob_df.columns = self.measurement_vars
        
        fig = LCAPlots.plot_profiles(prob_df, self.n_components)
        fig.show()

    def _render_distribution(self):
        """
        Internal method to render the class distribution bar chart.
        """
        if self.clusters is None:
            raise ValueError("No clusters found. Call .fit() first.")
            
        fig = LCAPlots.plot_clusters(self.clusters)
        fig.show()














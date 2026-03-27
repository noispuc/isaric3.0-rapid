import pickle
import pandas as pd
from pathlib import Path
from isaric.pipelines.LCA_phenotyping import RAPID_PhenotypeLCA

data_path = Path(__file__).parent.parent.parent.parent.parent / 'data' / 'child_1Wk_DengueInv.pkl'

def run_real_data_test():

    print(f"--- Loading real data from {data_path} ---")
    
    # 1. LOAD DATA AND VARIABLE LIST
    # According to Notebook 3, the pickle contains [dataframe, varList]
    try:
        with open(data_path, 'rb') as file:
            df = pickle.load(file)
            var_list = pickle.load(file)
    except FileNotFoundError:
        print(f"Error: file {data_path} not found. Check the path.")
        return

    print(f"Data loaded: {df.shape[0]} rows and {len(var_list)} clinical variables.")

    # 2. INITIALIZE PIPELINE
    # Setting K=3 for a quick validation run
    # 'HOSPITALIZ' was the structural variable used in the notebooks
    lca_pipeline = RAPID_PhenotypeLCA(
        data=df, 
        measurement_vars=var_list, 
        structural_var='HOSPITALIZ', 
        n_components=3
    )
    
    # 3. EXECUTE VALIDATION (Grid Search logic from Notebook 3)
    # Testing a small range for speed: 3 to 6 classes
    print("\nStep 1: Running Model Selection (Validation)...")
    grid_results = lca_pipeline._validation(range(3, 7))
    print("Grid Search Metrics (AIC/BIC):")
    print(grid_results)

    # 4. FIT FINAL MODEL (Logic from Notebook 2)
    print("\nStep 2: Fitting final model with K=3...")
    lca_pipeline.fit()
    
    # 5. GENERATE VISUALIZATIONS
    print("\nStep 3: Rendering Plotly reports...")
    # summary() calls _visualization(), which shows:
    # - Profiles Heatmap
    # - Class Distribution
    # - Grid Search Elbow Plot (since we ran _validation)
    lca_pipeline.summary()

    # 6. EXPORT RESULTS
    output_model = 'lca_model_real_data.pkl'
    lca_pipeline.save_model(output_model)
    print(f"\nSuccess! Results generated and model saved as {output_model}")

if __name__ == "__main__":
    run_real_data_test()
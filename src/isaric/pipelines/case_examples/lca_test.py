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
            data = pickle.load(file)
            if isinstance(data, (list, tuple)) and len(data) == 2:
                df, var_list = data
            else:
                df = data
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
    
    # 3. EXECUTE VALIDATION (Grid Search logic from Notebook 3/AdjGridSearch5)
    print("\nStep 1: Running Comprehensive Model Selection (Grid Search 3 to 6 classes)...")
    grid_results = lca_pipeline.grid_search(range(3, 7))
    print("Grid Search Metrics:")
    
    # Just displaying some important columns since it is comprehensive
    display_cols = ['ncomp', 'LL', 'AIC', 'BIC', 'entropy', 'relative_entropy']
    if 'p_value' in grid_results.columns:
        display_cols.append('p_value')
        
    print(grid_results[display_cols].to_string(index=False))

    # 4. SHOW GRID SEARCH PLOTS
    print("\nStep 2: Rendering Grid Search evaluation plots...")
    lca_pipeline.summary_grid_plots()
    
    # 5. SELECT MODEL & EXPLORE
    print("\nStep 3: Deciding on K=3 and exporting exploratory plots...")
    lca_pipeline.decide(3)
    lca_pipeline.describe()

    # 6. EXPORT RESULTS
    output_model = 'lca_model_real_data.pkl'
    lca_pipeline.save_model(output_model)
    print(f"\nSuccess! Results generated and model saved as {output_model}")

if __name__ == "__main__":
    run_real_data_test()
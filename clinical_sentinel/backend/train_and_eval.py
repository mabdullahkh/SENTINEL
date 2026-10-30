import os
from . import config
from . import database
from . import preprocessing
from . import models
from . import evaluation

def main():
    # 1. Train models and save preprocessor & thresholds
    manager = models.train_pipeline()
    
    # 2. Load dataset and preprocessor to run evaluation
    print("Loading data for evaluation...")
    df_raw = database.get_raw_modeling_data()
    prep = preprocessing.load_preprocessor()
    
    X = prep.transform(df_raw)
    y = df_raw[config.DISEASES]
    
    # 3. Run evaluation pipeline to generate metrics JSON
    evaluation.run_evaluation_pipeline(X, y, manager.models, manager.thresholds)
    
    print("\n✓ Pipeline training and evaluation complete!")

if __name__ == "__main__":
    main()

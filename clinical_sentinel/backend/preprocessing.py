import os
import pickle
import pandas as pd
import numpy as np
from . import config

PREPROCESSOR_PATH = os.path.join(config.DATA_DIR, "preprocessor.pkl")

class ClinicalPreprocessor:
    def __init__(self):
        self.medians = {}
        self.scalers = {}
        self.feature_cols = []
        self.categorical_cols = ['admission_type', 'insurance']
        self.numerical_cols = []
        self.dummy_cols = []

    def compute_clinical_scores(self, df):
        """Compute clinical scores like SOFA and SIRS components."""
        # SOFA Renal Score (based on creatinine_max)
        # 1: 1.2-1.9 mg/dL, 2: 2.0-3.4 mg/dL, 3: 3.5-4.9 mg/dL, 4: >5.0 mg/dL
        df['sofa_renal'] = 0
        df.loc[df['creatinine_max'] >= 1.2, 'sofa_renal'] = 1
        df.loc[df['creatinine_max'] >= 2.0, 'sofa_renal'] = 2
        df.loc[df['creatinine_max'] >= 3.5, 'sofa_renal'] = 3
        df.loc[df['creatinine_max'] >= 5.0, 'sofa_renal'] = 4

        # SOFA Coagulation Score (based on platelets_min)
        # 1: <150k, 2: <100k, 3: <50k, 4: <20k
        df['sofa_coag'] = 0
        df.loc[df['platelets_min'] < 150, 'sofa_coag'] = 1
        df.loc[df['platelets_min'] < 100, 'sofa_coag'] = 2
        df.loc[df['platelets_min'] < 50, 'sofa_coag'] = 3
        df.loc[df['platelets_min'] < 20, 'sofa_coag'] = 4

        # SIRS WBC Criteria (WBC > 12 or < 4)
        df['sirs_wbc'] = 0
        df.loc[(df['wbc_max'] > 12) | (df['wbc_min'] < 4), 'sirs_wbc'] = 1

        # BUN/Creatinine Ratio
        df['bun_creatinine_ratio'] = df['bun_median'] / (df['creatinine_median'] + 1e-5)
        
        return df

    def fit(self, df_raw):
        """Fit the preprocessor on the raw modeling dataframe."""
        df = df_raw.copy()

        # 1. Encode Gender (F -> 0, M -> 1)
        df['gender'] = df['gender'].map({'F': 0, 'M': 1}).fillna(0)

        # 2. Extract numeric columns from labs (min, max, median)
        lab_cols = [c for c in df.columns if any(suffix in c for suffix in ['_min', '_max', '_median'])]
        self.numerical_cols = ['age'] + lab_cols

        # 3. Compute cohort medians for imputation
        for col in self.numerical_cols:
            median_val = df[col].median()
            # If all are NaN, default to typical clinical values
            if np.isnan(median_val):
                if 'creatinine' in col: median_val = 1.0
                elif 'sodium' in col: median_val = 140.0
                elif 'potassium' in col: median_val = 4.0
                elif 'wbc' in col: median_val = 7.0
                elif 'platelets' in col: median_val = 250.0
                elif 'hemoglobin' in col: median_val = 13.0
                elif 'lactate' in col: median_val = 1.5
                elif 'glucose' in col: median_val = 100.0
                elif 'bun' in col: median_val = 15.0
                else: median_val = 0.0
            self.medians[col] = median_val
            df[col] = df[col].fillna(self.medians[col])

        # 4. Compute clinical scores (needs imputed labs to avoid NaNs)
        df = self.compute_clinical_scores(df)
        
        # Add engineered scores to numerical features
        self.numerical_cols += ['sofa_renal', 'sofa_coag', 'sirs_wbc', 'bun_creatinine_ratio']

        # 5. One-hot encode categorical features
        df_encoded = pd.get_dummies(df, columns=self.categorical_cols, drop_first=True)
        
        # Track dummy columns created
        self.dummy_cols = [c for c in df_encoded.columns if any(c.startswith(cat) for cat in self.categorical_cols)]

        # 6. Fit standard scalers for numerical columns
        for col in self.numerical_cols:
            mean = df_encoded[col].mean()
            std = df_encoded[col].std()
            self.scalers[col] = {'mean': mean, 'std': std if std > 0 else 1.0}

        # 7. Identify final features
        self.feature_cols = ['gender'] + self.numerical_cols + self.dummy_cols
        
        print(f"Preprocessor fit complete. Total features: {len(self.feature_cols)}")
        return self

    def transform(self, df_raw):
        """Transform raw data using fitted preprocessor settings."""
        df = df_raw.copy()

        # 1. Encode Gender
        df['gender'] = df['gender'].map({'F': 0, 'M': 1}).fillna(0)

        # 2. Impute missing numerical columns
        for col in self.medians:
            if col in df.columns:
                df[col] = df[col].fillna(self.medians[col])
            else:
                df[col] = self.medians[col]

        # 3. Compute clinical scores
        df = self.compute_clinical_scores(df)

        # 4. Scale numerical columns
        for col in self.numerical_cols:
            if col in df.columns:
                mean = self.scalers[col]['mean']
                std = self.scalers[col]['std']
                df[col] = (df[col] - mean) / std

        # 5. Handle categorical one-hot dummy variables manually to ensure exact alignment
        for dummy in self.dummy_cols:
            # Reconstruct categorical name and value
            # e.g., 'admission_type_URGENT' -> base = 'admission_type', value = 'URGENT'
            found = False
            for cat in self.categorical_cols:
                if dummy.startswith(cat + "_"):
                    base = cat
                    val = dummy[len(cat)+1:]
                    df[dummy] = (df[base] == val).astype(int)
                    found = True
                    break
            if not found:
                df[dummy] = 0

        # Fill any final NaNs
        df = df.fillna(0)

        # Cast to float to avoid dtype object issues in models (e.g. XGBoost)
        df_out = df[self.feature_cols].copy()
        for col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors='coerce').astype(float)
        return df_out


def save_preprocessor(preprocessor):
    """Save preprocessor object to disk."""
    with open(PREPROCESSOR_PATH, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"Preprocessor saved to {PREPROCESSOR_PATH}")

def load_preprocessor():
    """Load preprocessor object from disk."""
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Preprocessor file not found at {PREPROCESSOR_PATH}. Run training first.")
    with open(PREPROCESSOR_PATH, 'rb') as f:
        return pickle.load(f)

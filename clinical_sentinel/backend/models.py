import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, accuracy_score
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except (ImportError, Exception) as e:
    HAS_XGBOOST = False
    print(f"WARNING: XGBoost could not be loaded: {e}. Falling back to Scikit-Learn Gradient Boosting.")

from . import config
from . import database
from .preprocessing import ClinicalPreprocessor, save_preprocessor, load_preprocessor

# Try importing SHAP and LIME with fallback
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: SHAP not available, using fallback explainer.")

try:
    import lime
    import lime.lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False
    print("WARNING: LIME not available, using fallback explainer.")


class ModelManager:
    def __init__(self):
        self.models = {}  # Format: {disease: {model_type: trained_model}}
        self.thresholds = config.DEFAULT_THRESHOLDS.copy()
        self.feature_names = []

    def save(self):
        """Save models and thresholds to disk."""
        for disease, m_dict in self.models.items():
            for m_type, model in m_dict.items():
                path = os.path.join(config.MODELS_DIR, f"{disease}_{m_type}.pkl")
                with open(path, 'wb') as f:
                    pickle.dump(model, f)
        
        # Save thresholds
        thresh_path = os.path.join(config.MODELS_DIR, "thresholds.pkl")
        with open(thresh_path, 'wb') as f:
            pickle.dump(self.thresholds, f)
            
        print("All models and thresholds saved successfully.")

    def load(self):
        """Load models and thresholds from disk."""
        for disease in config.DISEASES:
            self.models[disease] = {}
            for m_type in config.MODEL_TYPES:
                path = os.path.join(config.MODELS_DIR, f"{disease}_{m_type}.pkl")
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        self.models[disease][m_type] = pickle.load(f)
                else:
                    raise FileNotFoundError(f"Model file not found at {path}. Please train first.")
        
        thresh_path = os.path.join(config.MODELS_DIR, "thresholds.pkl")
        if os.path.exists(thresh_path):
            with open(thresh_path, 'rb') as f:
                self.thresholds = pickle.load(f)
        
        # Try loading preprocessor to get feature names
        try:
            prep = load_preprocessor()
            self.feature_names = prep.feature_cols
        except Exception:
            pass
        
        print("All models and thresholds loaded successfully.")


def train_pipeline():
    """Extracts raw database data, preprocesses, trains XGBoost/RF, tunes thresholds, and saves objects."""
    print("\n" + "=" * 60)
    print("  Clinical Sentinel: Model Training Pipeline")
    print("=" * 60)
    
    # 1. Fetch data
    print("Loading raw clinical dataset from SQLite...")
    df_raw = database.get_raw_modeling_data()
    print(f"Loaded {len(df_raw)} records.")
    
    # 2. Fit Preprocessor
    print("Fitting clinical preprocessor and scaling features...")
    preprocessor = ClinicalPreprocessor()
    preprocessor.fit(df_raw)
    save_preprocessor(preprocessor)
    
    # Transform raw data
    X = preprocessor.transform(df_raw)
    y = df_raw[config.DISEASES]
    
    manager = ModelManager()
    manager.feature_names = preprocessor.feature_cols
    
    # 3. Train models per disease
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    
    for disease in config.DISEASES:
        print(f"\n--- Training Models for: {disease.upper()} ---")
        y_disease = y[disease]
        
        manager.models[disease] = {}
        
        # Check class balance
        pos_pct = (y_disease.sum() / len(y_disease)) * 100
        print(f"Class distribution: Positive = {y_disease.sum()} ({pos_pct:.2f}%), Negative = {len(y_disease) - y_disease.sum()}")
        
        for m_type in config.MODEL_TYPES:
            print(f"  Training {m_type.upper()}...")
            
            # Cross validation metrics accumulation
            cv_f1s = []
            cv_recalls = []
            cv_aucs = []
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_disease)):
                X_train, y_train = X.iloc[train_idx], y_disease.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y_disease.iloc[val_idx]
                
                if m_type == "random_forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE, class_weight='balanced', max_depth=8)
                else:  # xgboost
                    if HAS_XGBOOST:
                        scale_pos = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-5)
                        model = xgb.XGBClassifier(n_estimators=100, random_state=config.RANDOM_STATE, eval_metric='logloss', scale_pos_weight=scale_pos, max_depth=4, learning_rate=0.05)
                    else:
                        from sklearn.ensemble import GradientBoostingClassifier
                        model = GradientBoostingClassifier(n_estimators=100, random_state=config.RANDOM_STATE, max_depth=4, learning_rate=0.05)
                
                model.fit(X_train, y_train)
                
                # Validation evaluation at default threshold 0.5
                probs = model.predict_proba(X_val)[:, 1]
                preds = (probs >= 0.5).astype(int)
                
                cv_f1s.append(f1_score(y_val, preds, zero_division=0))
                cv_recalls.append(recall_score(y_val, preds, zero_division=0))
                cv_aucs.append(roc_auc_score(y_val, probs))
                
            print(f"    CV (Default Thresh 0.5) | AUC-ROC: {np.mean(cv_aucs):.3f} | F1: {np.mean(cv_f1s):.3f} | Recall: {np.mean(cv_recalls):.3f}")
            
            # Fit final model on all data
            if m_type == "random_forest":
                final_model = RandomForestClassifier(n_estimators=120, random_state=config.RANDOM_STATE, class_weight='balanced', max_depth=8)
            else:
                if HAS_XGBOOST:
                    scale_pos = (len(y_disease) - y_disease.sum()) / (y_disease.sum() + 1e-5)
                    final_model = xgb.XGBClassifier(n_estimators=120, random_state=config.RANDOM_STATE, eval_metric='logloss', scale_pos_weight=scale_pos, max_depth=4, learning_rate=0.05)
                else:
                    from sklearn.ensemble import GradientBoostingClassifier
                    final_model = GradientBoostingClassifier(n_estimators=120, random_state=config.RANDOM_STATE, max_depth=4, learning_rate=0.05)
                
            final_model.fit(X, y_disease)
            manager.models[disease][m_type] = final_model
            
        # 4. Tune decision thresholds for XGBoost / Gradient Boosting to balance F1 & Recall
        primary_model = manager.models[disease]['xgboost']
        probs_all = primary_model.predict_proba(X)[:, 1]
        
        best_threshold = 0.5
        best_score = 0.0
        
        # Grid search threshold to find maximum F1-score
        for thresh in np.linspace(0.1, 0.9, 81):
            preds = (probs_all >= thresh).astype(int)
            f1 = f1_score(y_disease, preds, zero_division=0)
            rec = recall_score(y_disease, preds, zero_division=0)
            prec = precision_score(y_disease, preds, zero_division=0)
            
            # We want to optimize F1 but ensure Recall is reasonably high (>= 0.70 if possible)
            score = f1 + 0.2 * rec # slightly favor recall for clinical safety
            
            if score > best_score and rec >= 0.65:
                best_score = score
                best_threshold = thresh
                
        manager.thresholds[disease] = float(best_threshold)
        print(f"  Optimized Threshold for XGBoost: {manager.thresholds[disease]:.2f}")
        
        final_preds = (probs_all >= manager.thresholds[disease]).astype(int)
        print(f"  Final metrics at tuned threshold | F1: {f1_score(y_disease, final_preds):.3f} | Recall: {recall_score(y_disease, final_preds):.3f} | Accuracy: {accuracy_score(y_disease, final_preds):.3f}")

    # 5. Save all models and preprocessors
    manager.save()
    print("=" * 60 + "\n")
    return manager


def generate_explanations(disease, patient_features_scaled, background_data_scaled=None, model_type="xgboost"):
    """
    Generates SHAP and LIME feature attribution explanation dicts.
    Returns:
        shap_exp (list of dicts): key feature names and their shap impact.
        lime_exp (list of dicts): key features and their local linear impact.
    """
    # Load model manager and threshold
    manager = ModelManager()
    manager.load()
    
    model = manager.models[disease][model_type]
    feature_names = manager.feature_names
    
    # 1. SHAP values
    shap_results = []
    if HAS_SHAP:
        try:
            # For tree models, TreeExplainer is fast
            if model_type == "random_forest" or (HAS_XGBOOST and isinstance(model, xgb.XGBClassifier)):
                # Use a small background sample if available to construct explainer
                if background_data_scaled is not None:
                    # sample 50 background rows
                    bg_sample = background_data_scaled.head(50)
                    explainer = shap.TreeExplainer(model, bg_sample)
                else:
                    explainer = shap.TreeExplainer(model)
                
                shap_values = explainer.shap_values(patient_features_scaled)
                
                # Handle shapes for binary classifiers
                if isinstance(shap_values, list): # RF sometimes returns list for each class
                    # Use class 1 shap values
                    shap_val = shap_values[1][0]
                elif len(shap_values.shape) == 3: # new shap versions
                    shap_val = shap_values[0, :, 1]
                else:
                    shap_val = shap_values[0]
                    
                # Pair with feature names
                for name, val in zip(feature_names, shap_val):
                    shap_results.append({"feature": name, "shap_value": float(val)})
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            
    # Fallback SHAP (feature importance * scaled value difference from mean)
    if not shap_results:
        try:
            importances = model.feature_importances_
            # approximate: contribution is importance * feature_value
            # features are already standardized, so value reflects dev from mean!
            val_vector = patient_features_scaled.iloc[0].values
            contribs = importances * val_vector
            for name, val in zip(feature_names, contribs):
                shap_results.append({"feature": name, "shap_value": float(val)})
        except Exception:
            pass
            
    # Sort shap results by absolute value
    shap_results = sorted(shap_results, key=lambda x: abs(x['shap_value']), reverse=True)[:8]

    # 2. LIME explanation
    lime_results = []
    if HAS_LIME and background_data_scaled is not None:
        try:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.array(background_data_scaled),
                feature_names=feature_names,
                class_names=['Low Risk', 'High Risk'],
                mode='classification'
            )
            exp = explainer.explain_instance(
                data_row=patient_features_scaled.iloc[0],
                predict_fn=model.predict_proba,
                num_features=8
            )
            # LIME returns list of tuples: (feature_rule, weight)
            for rule, weight in exp.as_list():
                # Extract clean feature name from rule
                feat_name = rule
                for name in feature_names:
                    if name in rule:
                        feat_name = name
                        break
                lime_results.append({"feature": feat_name, "rule": rule, "weight": float(weight)})
        except Exception as e:
            print(f"LIME explanation failed: {e}")
            
    if not lime_results:
        # Fallback LIME: simpler correlation/importance representation
        # Mock rules based on importance and feature sign
        try:
            importances = model.feature_importances_
            val_vector = patient_features_scaled.iloc[0].values
            for i, name in enumerate(feature_names):
                weight = float(importances[i] * np.sign(val_vector[i]))
                rule = f"{name} is {'high' if val_vector[i] > 0 else 'low'} ({val_vector[i]:.2f})"
                lime_results.append({"feature": name, "rule": rule, "weight": weight})
            lime_results = sorted(lime_results, key=lambda x: abs(x['weight']), reverse=True)[:8]
        except Exception:
            pass

    return shap_results, lime_results

if __name__ == "__main__":
    train_pipeline()

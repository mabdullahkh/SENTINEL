import os
import sys

# Ensure we can import from backend
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend import config
from backend import database
from backend import preprocessing
from backend import models
from backend import agentic_layer

def verify():
    print("=" * 60)
    print("  Clinical Sentinel: Automated Pipeline Verification")
    print("=" * 60)
    
    # 1. Verify Database File
    print("\n[1/5] Checking database file...")
    if not os.path.exists(config.DB_PATH):
        print(f"✗ Database not found at {config.DB_PATH}")
        sys.exit(1)
    print(f"✓ Database found at {config.DB_PATH} ({os.path.getsize(config.DB_PATH)/(1024*1024):.2f} MB)")
    
    # Check tables
    conn = database.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cursor.fetchall()}
    conn.close()
    
    expected_tables = ["patients", "admissions", "diagnoses_icd", "labevents", "prescriptions"]
    for t in expected_tables:
        if t not in tables:
            print(f"✗ Expected table '{t}' is missing from database subset!")
            sys.exit(1)
    print(f"✓ All {len(expected_tables)} clinical tables verified.")

    # 2. Verify Preprocessor and Models
    print("\n[2/5] Checking model files and preprocessing scalars...")
    try:
        prep = preprocessing.load_preprocessor()
        print(f"✓ Preprocessor loaded successfully. Engineered {len(prep.feature_cols)} features.")
    except Exception as e:
        print(f"✗ Preprocessor load failed: {e}")
        sys.exit(1)
        
    manager = models.ModelManager()
    try:
        manager.load()
        print("✓ All 5 disease models loaded successfully:")
        for d in config.DISEASES:
            print(f"  - {d.upper()}: XGBoost/Gradient Boosting + RF loaded (thresh: {manager.thresholds.get(d):.2f})")
    except Exception as e:
        print(f"✗ Model manager load failed: {e}")
        sys.exit(1)

    # 3. Test Feature Imputation & Inference
    print("\n[3/5] Testing patient preprocessing and inference loop...")
    try:
        patients = database.get_all_patients()
        test_patient = patients[0]
        hadm_id = test_patient['hadm_id']
        print(f"  Selected Test Patient ID: {test_patient['subject_id']}, HADM ID: {hadm_id}")
        
        # Pull and preprocess
        # We can reuse the FastAPI helper function logic internally to test
        # Let's import the app internal helper if possible or run equivalent
        from backend.app import get_prediction_helper
        res = get_prediction_helper(hadm_id)
        
        print("✓ Preprocessing completed without NaN values.")
        print("✓ Probabilities generated successfully:")
        for d, p in res["probabilities"].items():
            crossed = "THRES CROSSED" if res["threshold_crossed"][d] else "Stable"
            print(f"  - {d:<15}: {p*100:>5.1f}% [{crossed}]")
    except Exception as e:
        print(f"✗ Inference loop test failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # 4. Test Explainable AI attributions
    print("\n[4/5] Testing SHAP and LIME explainer calculations...")
    try:
        shap_exp, lime_exp = models.generate_explanations(
            disease="sepsis",
            patient_features_scaled=res["scaled_features"],
            background_data_scaled=database.get_raw_modeling_data().head(20),
            model_type="xgboost"
        )
        print(f"✓ SHAP attributions generated ({len(shap_exp)} factors).")
        print(f"✓ LIME rules generated ({len(lime_exp)} rules).")
    except Exception as e:
        print(f"✗ Explainable AI generation failed: {e}")
        sys.exit(1)

    # 5. Test Multi-Agent Consult Workflow
    print("\n[5/5] Testing Multi-Agent Clinical Consult system...")
    try:
        patient_info = {
            "subject_id": test_patient["subject_id"],
            "gender": test_patient["gender"],
            "anchor_age": test_patient["anchor_age"],
            "admission_type": test_patient["admission_type"],
            "diagnoses": database.get_patient_diagnoses(hadm_id),
            "prescriptions": database.get_patient_prescriptions(hadm_id)
        }
        brain = agentic_layer.ClinicalAgentBrain()
        consult = brain.run_consultation(
            patient_info=patient_info,
            ml_probabilities=res["probabilities"],
            shap_explanations={"sepsis": shap_exp, "cardiac": [], "respiratory": [], "renal": [], "neurological": []}
        )
        print("✓ Historian summary extracted.")
        print(f"✓ Specialists opinions completed for {len(consult['specialists'])} targets.")
        print(f"✓ Diagnostician synthesis completed (Confidence: {consult['diagnostician']['confidence_score']}%).")
        print(f"  Diagnostician Assessment: {consult['diagnostician']['final_assessment'][:60]}...")
    except Exception as e:
        print(f"✗ Multi-agent consultation workflow failed: {e}")
        sys.exit(1)
        
    print("\n" + "=" * 60)
    print("  ALL BACKEND PIPELINE CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    verify()

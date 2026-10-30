import os
import json
import math
import traceback
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Union

from . import config
from . import database
from . import preprocessing
from . import models
from . import agentic_layer
from . import evaluation


def sanitize_floats(obj):
    """Recursively replace NaN/Inf float values with None for JSON compliance."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_floats(v) for v in obj]
    return obj

app = FastAPI(title="Clinical Sentinel: EHR & Multi-Agent Intelligence")

# Paths
FRONTEND_DIR = os.path.join(os.path.dirname(config.BACKEND_DIR), "frontend")
os.makedirs(FRONTEND_DIR, exist_ok=True)

# Mount static files at /static
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class NewPatientLabs(BaseModel):
    creatinine: Optional[float] = None
    bun: Optional[float] = None
    sodium: Optional[float] = None
    potassium: Optional[float] = None
    wbc: Optional[float] = None
    platelets: Optional[float] = None
    hemoglobin: Optional[float] = None
    lactate: Optional[float] = None
    glucose: Optional[float] = None


class NewPatientRecord(BaseModel):
    gender: str = "F"
    anchor_age: int = 50
    admission_type: str = "EW EMER."
    insurance: str = "Medicare"
    primary_diagnosis: Optional[str] = "Unspecified Observation"
    labs: NewPatientLabs


class AddReportRequest(BaseModel):
    hadm_id: Optional[Union[int, str]] = None
    admission_type: Optional[str] = "EW EMER."
    insurance: Optional[str] = "Medicare"
    primary_diagnosis: Optional[str] = ""
    labs: NewPatientLabs


class ClinicianFeedback(BaseModel):
    hadm_id: int
    disease: str
    clinician_action: str  # e.g., 'agree', 'override_high_risk', 'override_low_risk'
    notes: Optional[str] = ""


# Load ML elements at startup helper
def get_prediction_helper(hadm_id: int):
    """Internal helper to calculate risk scores for an admission."""
    # 1. Fetch patient admission details
    conn = database.get_connection()
    q_adm = """
        SELECT a.hadm_id, a.subject_id, p.gender, p.anchor_age as age, a.admission_type, a.insurance
        FROM admissions a
        JOIN patients p ON a.subject_id = p.subject_id
        WHERE a.hadm_id = ?
    """
    df_cohort = pd.read_sql_query(q_adm, conn, params=(hadm_id,))
    
    if df_cohort.empty:
        conn.close()
        raise HTTPException(status_code=404, detail="Admission not found")
        
    # 2. Fetch and aggregate labs
    lab_items = {
        50912: "creatinine",
        51006: "bun",
        50983: "sodium",
        50971: "potassium",
        51301: "wbc",
        51265: "platelets",
        51222: "hemoglobin",
        50813: "lactate",
        50931: "glucose"
    }
    placeholders = ",".join(map(str, lab_items.keys()))
    q_labs = f"""
        SELECT itemid, valuenum
        FROM labevents
        WHERE hadm_id = ? AND itemid IN ({placeholders}) AND valuenum IS NOT NULL
    """
    df_labs = pd.read_sql_query(q_labs, conn, params=(hadm_id,))
    conn.close()
    
    # Construct base feature dataframe
    df_raw = df_cohort.copy()
    
    # Aggregate labs for this patient
    if df_labs.empty:
        for name in lab_items.values():
            df_raw[f"{name}_min"] = None
            df_raw[f"{name}_max"] = None
            df_raw[f"{name}_median"] = None
    else:
        df_labs['lab_name'] = df_labs['itemid'].map(lab_items)
        lab_agg = df_labs.groupby('lab_name')['valuenum'].agg(['min', 'max', 'median'])
        for name in lab_items.values():
            if name in lab_agg.index:
                df_raw[f"{name}_min"] = lab_agg.loc[name, 'min']
                df_raw[f"{name}_max"] = lab_agg.loc[name, 'max']
                df_raw[f"{name}_median"] = lab_agg.loc[name, 'median']
            else:
                df_raw[f"{name}_min"] = None
                df_raw[f"{name}_max"] = None
                df_raw[f"{name}_median"] = None

    # Load preprocessor
    prep = preprocessing.load_preprocessor()
    X_scaled = prep.transform(df_raw)
    
    # Load Models and Thresholds
    manager = models.ModelManager()
    manager.load()
    
    # Run predictions
    probabilities = {}
    threshold_crossed = {}
    
    for disease in config.DISEASES:
        model = manager.models[disease]['xgboost']
        prob = float(model.predict_proba(X_scaled)[0, 1])
        thresh = manager.thresholds.get(disease, 0.5)
        
        probabilities[disease] = prob
        threshold_crossed[disease] = bool(prob >= thresh)
        
    return {
        "hadm_id": hadm_id,
        "probabilities": probabilities,
        "thresholds": manager.thresholds,
        "threshold_crossed": threshold_crossed,
        "raw_features": df_raw.iloc[0].to_dict(),
        "scaled_features": X_scaled
    }


@app.get("/")
def get_index():
    """Serve the primary HTML UI dashboard."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Clinical Sentinel Web UI: HTML file is being initialized."}


@app.get("/results")
def get_results():
    """Serve the research results & evaluation metrics page."""
    results_path = os.path.join(FRONTEND_DIR, "results.html")
    if os.path.exists(results_path):
        return FileResponse(results_path)
    return {"message": "Results page not found."}


@app.get("/add-patient")
def get_add_patient_page():
    """Serve the Add New Patient UI page."""
    add_path = os.path.join(FRONTEND_DIR, "add_patient.html")
    if os.path.exists(add_path):
        return FileResponse(add_path)
    return {"message": "Add patient page not found."}


@app.get("/patient-history")
def get_patient_history_page():
    """Serve the Patient History & Timeline Inspector UI page."""
    hist_path = os.path.join(FRONTEND_DIR, "patient_history.html")
    if os.path.exists(hist_path):
        return FileResponse(hist_path)
    return {"message": "Patient history page not found."}


@app.get("/api/patient/{subject_id}/history")
def api_get_patient_history(subject_id: int):
    """Retrieve full chronological visit and lab history for an existing patient."""
    try:
        profile = database.get_patient_profile(subject_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Patient not found")
            
        admissions = database.get_patient_admissions(subject_id)
        
        # Populate each admission with diagnoses, prescriptions, labs, AND predictions
        for adm in admissions:
            hadm_id = adm['hadm_id']
            adm['diagnoses'] = database.get_patient_diagnoses(hadm_id)
            adm['prescriptions'] = database.get_patient_prescriptions(hadm_id)
            adm['labs'] = database.get_patient_labs(hadm_id)
            
            # Compute risk scores if labs exist for this admission
            try:
                pred_res = get_prediction_helper(hadm_id)
                adm['predictions'] = {
                    "probabilities": pred_res["probabilities"],
                    "thresholds": pred_res["thresholds"],
                    "threshold_crossed": pred_res["threshold_crossed"]
                }
            except Exception:
                adm['predictions'] = None
            
        return {
            "status": "success",
            "profile": profile,
            "admissions": admissions
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "trace": traceback.format_exc()}


@app.post("/api/patient/{subject_id}/add-report")
def api_post_add_report(subject_id: int, report_data: AddReportRequest):
    """Append a new lab report or log a new hospital visit for an existing patient."""
    try:
        data_dict = report_data.dict()
        res = database.add_report_for_existing_patient(subject_id, data_dict)
        hadm_id = res["hadm_id"]
        
        # Recalculate ML predictions for the target admission
        pred_res = get_prediction_helper(hadm_id)
        
        return {
            "status": "success",
            "message": res["message"],
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "predictions": {
                "probabilities": pred_res["probabilities"],
                "thresholds": pred_res["thresholds"],
                "threshold_crossed": pred_res["threshold_crossed"]
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "trace": traceback.format_exc()}


@app.post("/api/patient/add")
def api_post_add_patient(patient_data: NewPatientRecord):
    """Insert a new patient record into SQLite database and calculate predictions immediately."""
    try:
        data_dict = patient_data.dict()
        res = database.add_new_patient_record(data_dict)
        hadm_id = res["hadm_id"]
        
        # Calculate instant ML predictions for the newly added patient
        pred_res = get_prediction_helper(hadm_id)
        
        return {
            "status": "success",
            "message": res["message"],
            "subject_id": res["subject_id"],
            "hadm_id": hadm_id,
            "predictions": {
                "probabilities": pred_res["probabilities"],
                "thresholds": pred_res["thresholds"],
                "threshold_crossed": pred_res["threshold_crossed"]
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "trace": traceback.format_exc()}


@app.get("/api/patients")
def api_get_patients():
    """Get list of patients in cohort."""
    try:
        patients = database.get_all_patients()
        return {"status": "success", "count": len(patients), "data": patients}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/patient/{subject_id}")
def api_get_patient_detail(subject_id: int):
    """Retrieve detailed clinical profile of a patient."""
    try:
        profile = database.get_patient_profile(subject_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Patient not found")
            
        admissions = database.get_patient_admissions(subject_id)
        
        # Populate each admission with diagnoses, meds, labs
        for adm in admissions:
            hadm_id = adm['hadm_id']
            adm['diagnoses'] = database.get_patient_diagnoses(hadm_id)
            adm['prescriptions'] = database.get_patient_prescriptions(hadm_id)
            adm['labs'] = database.get_patient_labs(hadm_id)
            
        return {"status": "success", "profile": profile, "admissions": admissions}
    except Exception as e:
        return {"status": "error", "message": str(e), "trace": traceback.format_exc()}


@app.get("/api/patient/{hadm_id}/scores")
def api_get_patient_scores(hadm_id: int):
    """Compute and return ML predictions for the 5 target diseases."""
    try:
        res = get_prediction_helper(hadm_id)
        return {
            "status": "success",
            "hadm_id": res["hadm_id"],
            "probabilities": res["probabilities"],
            "thresholds": res["thresholds"],
            "threshold_crossed": res["threshold_crossed"],
            "raw_labs": {k: v for k, v in res["raw_features"].items() if any(s in k for s in ['_min', '_max', '_median'])}
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "trace": traceback.format_exc()}


@app.get("/api/patient/{hadm_id}/explain")
def api_get_patient_explanations(hadm_id: int):
    """Retrieve SHAP and LIME feature explanations for the patient."""
    try:
        # Get pred helper (contains scaled features)
        pred_res = get_prediction_helper(hadm_id)
        X_scaled = pred_res["scaled_features"]
        
        # Get background sample for LIME/SHAP
        raw_background = database.get_raw_modeling_data().head(100)
        prep = preprocessing.load_preprocessor()
        bg_scaled = prep.transform(raw_background)
        
        explanations = {}
        for disease in config.DISEASES:
            shap_exp, lime_exp = models.generate_explanations(
                disease=disease,
                patient_features_scaled=X_scaled,
                background_data_scaled=bg_scaled,
                model_type="xgboost"
            )
            explanations[disease] = {
                "shap": shap_exp,
                "lime": lime_exp
            }
            
        return {"status": "success", "hadm_id": hadm_id, "explanations": explanations}
    except Exception as e:
        return {"status": "error", "message": str(e), "trace": traceback.format_exc()}


@app.post("/api/patient/{hadm_id}/consult")
def api_post_patient_consult(hadm_id: int):
    """Run the multi-agent clinical panel consultation."""
    try:
        # 1. Fetch patient profile details
        conn = database.get_connection()
        q_adm = """
            SELECT a.hadm_id, a.subject_id, p.gender, p.anchor_age, a.admission_type, a.insurance
            FROM admissions a
            JOIN patients p ON a.subject_id = p.subject_id
            WHERE a.hadm_id = ?
        """
        df_cohort = pd.read_sql_query(q_adm, conn, params=(hadm_id,))
        if df_cohort.empty:
            conn.close()
            raise HTTPException(status_code=404, detail="Admission not found")
        
        patient_info = df_cohort.iloc[0].to_dict()
        patient_info['diagnoses'] = database.get_patient_diagnoses(hadm_id)
        patient_info['prescriptions'] = database.get_patient_prescriptions(hadm_id)
        conn.close()
        
        # 2. Get predictions
        pred_res = get_prediction_helper(hadm_id)
        probs = pred_res["probabilities"]
        X_scaled = pred_res["scaled_features"]
        
        # 3. Compute SHAP explanations (to feed to specialists)
        raw_background = database.get_raw_modeling_data().head(100)
        prep = preprocessing.load_preprocessor()
        bg_scaled = prep.transform(raw_background)
        
        shap_exps = {}
        for disease in config.DISEASES:
            shap_exp, _ = models.generate_explanations(
                disease=disease,
                patient_features_scaled=X_scaled,
                background_data_scaled=bg_scaled,
                model_type="xgboost"
            )
            shap_exps[disease] = shap_exp

        # 4. Trigger Multi-Agent Consultation
        brain = agentic_layer.ClinicalAgentBrain()
        consult_res = brain.run_consultation(
            patient_info=patient_info,
            ml_probabilities=probs,
            shap_explanations=shap_exps
        )
        
        # Log audit log
        audit_log_path = os.path.join(config.DATA_DIR, "audit_trail.jsonl")
        audit_entry = {
            "hadm_id": hadm_id,
            "subject_id": patient_info["subject_id"],
            "probabilities": probs,
            "diagnostician_score": consult_res["diagnostician"]["confidence_score"],
            "diagnostician_action": consult_res["diagnostician"]["final_assessment"]
        }
        with open(audit_log_path, 'a') as f:
            f.write(json.dumps(audit_entry) + "\n")
            
        return {
            "status": "success",
            "hadm_id": hadm_id,
            "consultation": consult_res
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "trace": traceback.format_exc()}


@app.get("/api/metrics")
def api_get_metrics():
    """Retrieve Saved System & Model evaluation reports."""
    try:
        metrics = evaluation.get_stored_metrics()
        return {"status": "success", "metrics": sanitize_floats(metrics)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/feedback")
def api_post_feedback(feedback: ClinicianFeedback):
    """Record clinician feedback/override for continuous training audit trail."""
    try:
        feedback_path = os.path.join(config.DATA_DIR, "clinician_feedback.csv")
        file_exists = os.path.exists(feedback_path)
        
        df = pd.DataFrame([feedback.dict()])
        df.to_csv(feedback_path, mode='a', header=not file_exists, index=False)
        
        return {"status": "success", "message": "Feedback recorded successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

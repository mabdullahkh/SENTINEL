import sqlite3
import math
import pandas as pd
import numpy as np
from . import config


def _clean_record(record: dict) -> dict:
    """Replace NaN/Inf float values with None so records are JSON-safe."""
    cleaned = {}
    for k, v in record.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            cleaned[k] = None
        else:
            cleaned[k] = v
    return cleaned


def df_to_records(df: pd.DataFrame) -> list:
    """Convert a DataFrame to a list of JSON-safe dicts (NaN → None)."""
    return [_clean_record(r) for r in df.to_dict(orient="records")]

def get_connection():
    """Establish connection to the SQLite database."""
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_all_patients():
    """Retrieve all patients in the cohort with key details."""
    conn = get_connection()
    query = """
        SELECT DISTINCT 
            p.subject_id, 
            p.gender, 
            p.anchor_age,
            a.hadm_id,
            a.admission_type,
            a.insurance,
            a.admittime,
            a.hospital_expire_flag
        FROM patients p
        JOIN admissions a ON p.subject_id = a.subject_id
        ORDER BY p.subject_id
    """
    df = pd.read_sql_query(query, conn)
    
    # Add primary diagnosis string for display
    diag_query = """
        SELECT d.hadm_id, d.icd_code, lookup.long_title
        FROM diagnoses_icd d
        JOIN d_icd_diagnoses lookup ON d.icd_code = lookup.icd_code AND d.icd_version = lookup.icd_version
        WHERE d.seq_num = 1
    """
    diags = pd.read_sql_query(diag_query, conn)
    df = df.merge(diags, on="hadm_id", how="left")
    conn.close()
    return df_to_records(df)

def get_patient_profile(subject_id):
    """Retrieve details for a single patient."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT subject_id, gender, anchor_age, anchor_year, dod
        FROM patients
        WHERE subject_id = ?
    """, (subject_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def get_patient_admissions(subject_id):
    """Get all admissions for a patient."""
    conn = get_connection()
    query = """
        SELECT hadm_id, admittime, dischtime, deathtime, admission_type, insurance, discharge_location, hospital_expire_flag
        FROM admissions
        WHERE subject_id = ?
        ORDER BY admittime
    """
    df = pd.read_sql_query(query, conn, params=(subject_id,))
    conn.close()
    return df_to_records(df)

def get_patient_diagnoses(hadm_id):
    """Get diagnoses for a specific admission."""
    conn = get_connection()
    query = """
        SELECT d.icd_code, d.icd_version, d.seq_num, lookup.long_title
        FROM diagnoses_icd d
        JOIN d_icd_diagnoses lookup ON d.icd_code = lookup.icd_code AND d.icd_version = lookup.icd_version
        WHERE d.hadm_id = ?
        ORDER BY d.seq_num
    """
    df = pd.read_sql_query(query, conn, params=(hadm_id,))
    conn.close()
    return df_to_records(df)

def get_patient_prescriptions(hadm_id):
    """Get prescriptions for an admission."""
    conn = get_connection()
    query = """
        SELECT starttime, stoptime, drug, drug_type, prod_strength, route
        FROM prescriptions
        WHERE hadm_id = ?
        ORDER BY starttime
    """
    df = pd.read_sql_query(query, conn, params=(hadm_id,))
    conn.close()
    return df_to_records(df)

def get_patient_labs(hadm_id):
    """Get lab measurements for an admission."""
    conn = get_connection()
    query = """
        SELECT l.charttime, l.itemid, d.label, l.valuenum, l.valueuom, l.ref_range_lower, l.ref_range_upper, l.flag
        FROM labevents l
        JOIN d_labitems d ON l.itemid = d.itemid
        WHERE l.hadm_id = ?
        ORDER BY l.charttime
    """
    df = pd.read_sql_query(query, conn, params=(hadm_id,))
    conn.close()
    return df_to_records(df)

def get_raw_modeling_data():
    """
    Extracts flat patient admission features and targets for training ML models.
    Returns:
        X (DataFrame): demographics + aggregated lab features.
        y (DataFrame): binary targets for the 5 diseases.
    """
    conn = get_connection()
    
    # 1. Base cohort (admissions + demographics)
    cohort_query = """
        SELECT 
            a.subject_id,
            a.hadm_id,
            p.gender,
            p.anchor_age as age,
            a.admission_type,
            a.insurance
        FROM admissions a
        JOIN patients p ON a.subject_id = p.subject_id
    """
    df_cohort = pd.read_sql_query(cohort_query, conn)
    
    # 2. Extract targets (binary labels for 5 diseases based on diagnoses)
    diag_query = "SELECT hadm_id, icd_code FROM diagnoses_icd"
    df_diags = pd.read_sql_query(diag_query, conn)
    
    # Create disease label mappings
    targets = {}
    for disease, prefixes in config.DISEASE_ICD_PREFIXES.items():
        # Match if code starts with any of the prefixes
        # Note: strip spaces and standardise codes
        df_diags[disease] = df_diags['icd_code'].apply(
            lambda x: 1 if x and any(str(x).startswith(prefix) for prefix in prefixes) else 0
        )
        # Aggregate by hadm_id (if any code matches, patient has the disease in that admission)
        targets[disease] = df_diags.groupby('hadm_id')[disease].max()
        
    df_targets = pd.DataFrame(targets).reset_index()
    
    # Merge cohort and targets
    df_data = df_cohort.merge(df_targets, on="hadm_id", how="left").fillna(0)
    
    # 3. Aggregate Labs for features
    # Common clinical lab item IDs:
    # 50912: Creatinine
    # 51006: BUN
    # 50983: Sodium
    # 50971: Potassium
    # 51301: WBC
    # 51265: Platelets
    # 51222: Hemoglobin (Hematology)
    # 50811: Hemoglobin (Blood Gas)
    # 50813: Lactate
    # 50931: Glucose
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
    lab_query = f"""
        SELECT hadm_id, itemid, valuenum
        FROM labevents
        WHERE itemid IN ({placeholders}) AND valuenum IS NOT NULL
    """
    df_labs = pd.read_sql_query(lab_query, conn)
    conn.close()
    
    # If no labs, return empty aggregations
    if df_labs.empty:
        # Create empty lab features
        for itemid, name in lab_items.items():
            df_data[f"{name}_min"] = np.nan
            df_data[f"{name}_max"] = np.nan
            df_data[f"{name}_median"] = np.nan
    else:
        # Map item IDs to names
        df_labs['lab_name'] = df_labs['itemid'].map(lab_items)
        
        # Group by hadm_id and lab_name, compute min, max, median
        lab_agg = df_labs.groupby(['hadm_id', 'lab_name'])['valuenum'].agg(['min', 'max', 'median']).unstack()
        
        # Flatten columns
        lab_agg.columns = [f"{col[1]}_{col[0]}" for col in lab_agg.columns]
        lab_agg = lab_agg.reset_index()
        
        # Merge labs into cohort
        df_data = df_data.merge(lab_agg, on="hadm_id", how="left")
        
    return df_data


def add_new_patient_record(data: dict) -> dict:
    """
    Inserts a new patient, admission, and associated lab events into the database.
    
    Expected data structure:
    {
        "gender": "M" or "F",
        "anchor_age": int,
        "admission_type": "EW EMER.", "URGENT", "ELECTIVE", etc.,
        "insurance": "Medicare", "Medicaid", "Private",
        "primary_diagnosis": optional str,
        "labs": {
            "creatinine": float,
            "bun": float,
            "sodium": float,
            "potassium": float,
            "wbc": float,
            "platelets": float,
            "hemoglobin": float,
            "lactate": float,
            "glucose": float
        }
    }
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Generate new subject_id
        cursor.execute("SELECT MAX(subject_id) FROM patients")
        max_sub = cursor.fetchone()[0] or 0
        new_subject_id = max(max_sub + 1, 90000001)
        
        # Generate new hadm_id
        cursor.execute("SELECT MAX(hadm_id) FROM admissions")
        max_adm = cursor.fetchone()[0] or 0
        new_hadm_id = max(max_adm + 1, 90000001)
        
        # 1. Insert patient
        gender = str(data.get("gender", "F")).upper()
        age = int(data.get("anchor_age", 50))
        cursor.execute("""
            INSERT INTO patients (subject_id, gender, anchor_age, anchor_year)
            VALUES (?, ?, ?, 2150)
        """, (new_subject_id, gender, age))
        
        # 2. Insert admission
        adm_type = str(data.get("admission_type", "EW EMER."))
        insurance = str(data.get("insurance", "Medicare"))
        cursor.execute("""
            INSERT INTO admissions (hadm_id, subject_id, admittime, admission_type, insurance, hospital_expire_flag)
            VALUES (?, ?, datetime('now'), ?, ?, 0)
        """, (new_hadm_id, new_subject_id, adm_type, insurance))
        
        # 3. Insert primary diagnosis if provided
        diag_title = data.get("primary_diagnosis", "").strip() or "Unspecified Observation"
        icd_code = "R69"
        cursor.execute("""
            INSERT OR IGNORE INTO d_icd_diagnoses (icd_code, icd_version, long_title)
            VALUES (?, 10, ?)
        """, (icd_code, diag_title))
        cursor.execute("""
            INSERT INTO diagnoses_icd (subject_id, hadm_id, seq_num, icd_code, icd_version)
            VALUES (?, ?, 1, ?, 10)
        """, (new_subject_id, new_hadm_id, icd_code))
        
        # 4. Insert labs into labevents
        cursor.execute("SELECT MAX(labevent_id) FROM labevents")
        max_lab_id = cursor.fetchone()[0] or 90000000
        current_lab_id = max(max_lab_id, 90000000)

        lab_item_map = {
            "creatinine": 50912,
            "bun": 51006,
            "sodium": 50983,
            "potassium": 50971,
            "wbc": 51301,
            "platelets": 51265,
            "hemoglobin": 51222,
            "lactate": 50813,
            "glucose": 50931
        }
        
        labs_data = data.get("labs", {})
        for lab_name, itemid in lab_item_map.items():
            val = labs_data.get(lab_name)
            if val is not None:
                try:
                    val_num = float(val)
                    current_lab_id += 1
                    cursor.execute("""
                        INSERT INTO labevents (labevent_id, subject_id, hadm_id, itemid, charttime, valuenum)
                        VALUES (?, ?, ?, ?, datetime('now'), ?)
                    """, (current_lab_id, new_subject_id, new_hadm_id, itemid, val_num))
                except (ValueError, TypeError):
                    pass

        conn.commit()
        conn.close()
        
        return {
            "status": "success",
            "subject_id": new_subject_id,
            "hadm_id": new_hadm_id,
            "message": f"Patient record successfully created with Subject ID {new_subject_id} and Admission ID {new_hadm_id}."
        }
    except Exception as e:
        conn.rollback()
        conn.close()
        raise e


def add_report_for_existing_patient(subject_id: int, data: dict) -> dict:
    """
    Appends a new lab report or logs a new hospital visit for an existing patient.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Check patient existence
        cursor.execute("SELECT subject_id FROM patients WHERE subject_id = ?", (subject_id,))
        if not cursor.fetchone():
            conn.close()
            raise ValueError(f"Patient with Subject ID {subject_id} not found in database.")
            
        target_hadm_id = data.get("hadm_id")
        
        # If no target hadm_id provided or explicitly requested new admission, create one
        if not target_hadm_id or str(target_hadm_id).lower() == "new":
            cursor.execute("SELECT MAX(hadm_id) FROM admissions")
            max_adm = cursor.fetchone()[0] or 0
            target_hadm_id = max(max_adm + 1, 90000001)
            
            adm_type = str(data.get("admission_type", "EW EMER."))
            insurance = str(data.get("insurance", "Medicare"))
            cursor.execute("""
                INSERT INTO admissions (hadm_id, subject_id, admittime, admission_type, insurance, hospital_expire_flag)
                VALUES (?, ?, datetime('now'), ?, ?, 0)
            """, (target_hadm_id, subject_id, adm_type, insurance))
            
            # Primary diagnosis if provided
            diag_title = data.get("primary_diagnosis", "").strip()
            if diag_title:
                icd_code = "R69"
                cursor.execute("""
                    INSERT OR IGNORE INTO d_icd_diagnoses (icd_code, icd_version, long_title)
                    VALUES (?, 10, ?)
                """, (icd_code, diag_title))
                cursor.execute("""
                    INSERT INTO diagnoses_icd (subject_id, hadm_id, seq_num, icd_code, icd_version)
                    VALUES (?, ?, 1, ?, 10)
                """, (subject_id, target_hadm_id, icd_code))
        else:
            target_hadm_id = int(target_hadm_id)
            
        # Insert labs into labevents
        cursor.execute("SELECT MAX(labevent_id) FROM labevents")
        max_lab_id = cursor.fetchone()[0] or 90000000
        current_lab_id = max(max_lab_id, 90000000)

        lab_item_map = {
            "creatinine": 50912,
            "bun": 51006,
            "sodium": 50983,
            "potassium": 50971,
            "wbc": 51301,
            "platelets": 51265,
            "hemoglobin": 51222,
            "lactate": 50813,
            "glucose": 50931
        }
        
        labs_data = data.get("labs", {})
        inserted_count = 0
        for lab_name, itemid in lab_item_map.items():
            val = labs_data.get(lab_name)
            if val is not None:
                try:
                    val_num = float(val)
                    current_lab_id += 1
                    cursor.execute("""
                        INSERT INTO labevents (labevent_id, subject_id, hadm_id, itemid, charttime, valuenum)
                        VALUES (?, ?, ?, ?, datetime('now'), ?)
                    """, (current_lab_id, subject_id, target_hadm_id, itemid, val_num))
                    inserted_count += 1
                except (ValueError, TypeError):
                    pass

        conn.commit()
        conn.close()
        
        return {
            "status": "success",
            "subject_id": subject_id,
            "hadm_id": target_hadm_id,
            "labs_inserted": inserted_count,
            "message": f"Successfully recorded {inserted_count} new lab biomarkers for Subject ID {subject_id} under Admission {target_hadm_id}."
        }
    except Exception as e:
        conn.rollback()
        conn.close()
        raise e



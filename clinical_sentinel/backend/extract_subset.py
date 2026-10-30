import sqlite3
import os
import time

# Configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
os.makedirs(DATA_DIR, exist_ok=True)

MAIN_DB = "/Users/muhammadabdullah/Downloads/SidraMsThesis/mimic-iv-3.1/xl files only/mimic_iv.db"
SUBSET_DB = os.path.join(DATA_DIR, "mimic_subset.db")

# Target diseases definitions (prefixes of ICD-9 and ICD-10 codes)
DISEASE_ICD_PREFIXES = {
    "sepsis": [
        "99591", "99592", "78552", "77181",  # ICD-9
        "A40", "A41", "R652"                 # ICD-10
    ],
    "cardiac": [
        "428", "410",                        # ICD-9 (Heart Failure, MI)
        "I50", "I21"                         # ICD-10
    ],
    "respiratory": [
        "486", "51881", "496",               # ICD-9 (Pneumonia, Resp Failure, COPD)
        "J18", "J96", "J44"                  # ICD-10
    ],
    "renal": [
        "584", "585",                        # ICD-9 (Acute Renal Failure, CKD)
        "N17", "N18"                         # ICD-10
    ],
    "neurological": [
        "434", "430", "431",                 # ICD-9 (Stroke, Hemorrhage)
        "I63", "I60", "I61"                  # ICD-10
    ]
}

def extract():
    print("=" * 60)
    print("  Clinical Sentinel: Cohort Extractor")
    print(f"  Source:      {MAIN_DB}")
    print(f"  Destination: {SUBSET_DB}")
    print("=" * 60)

    if os.path.exists(SUBSET_DB):
        print(f"Subset DB already exists at {SUBSET_DB}. Removing to re-extract...")
        os.remove(SUBSET_DB)

    # 1. Open connection to source database
    conn = sqlite3.connect(MAIN_DB)
    cursor = conn.cursor()

    # 2. Find subject_ids for each disease category
    cohort_subjects = set()
    print("\n[1/5] Sampling patient cohorts per disease category...")
    
    for disease, prefixes in DISEASE_ICD_PREFIXES.items():
        # Build SQL condition
        conditions = []
        for prefix in prefixes:
            conditions.append(f"icd_code LIKE '{prefix}%'")
        sql_cond = " OR ".join(conditions)
        
        query = f"""
            SELECT DISTINCT subject_id 
            FROM diagnoses_icd 
            WHERE {sql_cond} 
            LIMIT 600
        """
        cursor.execute(query)
        subjects = {row[0] for row in cursor.fetchall()}
        print(f"  - {disease.upper()}: found {len(subjects)} patients")
        cohort_subjects.update(subjects)

    cohort_list = list(cohort_subjects)
    print(f"Total unique patients in subset cohort: {len(cohort_list)}")
    
    # 3. Create subset DB tables and attach it to source connection
    print("\n[2/5] Creating subset database tables...")
    sub_conn = sqlite3.connect(SUBSET_DB)
    sub_cursor = sub_conn.cursor()
    
    # Copy DDL schemas from source
    tables_to_copy = [
        "patients", "admissions", "diagnoses_icd", "prescriptions", 
        "d_icd_diagnoses", "d_icd_procedures", "d_labitems"
    ]
    
    for table in tables_to_copy:
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'")
        ddl = cursor.fetchone()[0]
        sub_cursor.execute(ddl)
        
    # For labevents, we copy its schema too (use robust table definition)
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='labevents'")
    labevents_ddl = cursor.fetchone()[0]
    sub_cursor.execute(labevents_ddl)
    sub_conn.commit()
    sub_conn.close()

    # 4. Attach subset database to main connection for fast transfers
    cursor.execute(f"ATTACH DATABASE '{SUBSET_DB}' AS subset")
    
    # 5. Populate metadata / lookup tables (entirely, since they are small)
    print("\n[3/5] Copying dictionary lookup tables...")
    for table in ["d_icd_diagnoses", "d_icd_procedures", "d_labitems"]:
        t0 = time.time()
        cursor.execute(f"INSERT INTO subset.{table} SELECT * FROM main.{table}")
        print(f"  - {table} copied in {time.time()-t0:.2f}s")
        
    # 6. Copy clinical tables for selected cohort
    print("\n[4/5] Copying patient-level clinical records...")
    
    # We will write subject_ids into a temporary table in main DB to make JOINs fast
    cursor.execute("CREATE TEMP TABLE cohort_temp (subject_id INTEGER PRIMARY KEY)")
    cursor.executemany("INSERT OR IGNORE INTO cohort_temp VALUES (?)", [(s,) for s in cohort_list])
    conn.commit()

    patient_tables = ["patients", "admissions", "diagnoses_icd", "prescriptions"]
    for table in patient_tables:
        t0 = time.time()
        # Join with cohort_temp to filter
        cursor.execute(f"""
            INSERT INTO subset.{table}
            SELECT t.* FROM main.{table} t
            JOIN cohort_temp c ON t.subject_id = c.subject_id
        """)
        rows_copied = cursor.rowcount
        print(f"  - {table}: copied {rows_copied:,} rows in {time.time()-t0:.2f}s")
        
    # Copy labevents: This is the large table (~46M rows in original SQLite, 18GB CSV).
    # Since we join with cohort_temp (which is indexed and small), and if SQLite can use the temp table
    # or scan, let's see how long it takes.
    # To be extremely safe and monitor progress, we scan labevents in chunks using Python,
    # or let SQLite join. Let's do it using SQL join, as SQLite query planner will use the TEMP table primary key index
    # and perform an index lookup if it is indexed, but main.labevents doesn't have an index on subject_id.
    # Therefore, SQLite will do a single linear scan of main.labevents and join it with cohort_temp.
    # A single linear scan is very fast. Let's run it and time it!
    print("  - labevents: scanning and filtering (this may take 1-2 minutes)...")
    t0 = time.time()
    cursor.execute("""
        INSERT INTO subset.labevents
        SELECT t.* FROM main.labevents t
        JOIN cohort_temp c ON t.subject_id = c.subject_id
    """)
    rows_copied = cursor.rowcount
    print(f"  - labevents: copied {rows_copied:,} rows in {time.time()-t0:.2f}s")
    
    conn.commit()
    cursor.execute("DETACH DATABASE subset")
    conn.close()

    # 7. Create indexes in the subset database for high performance
    print("\n[5/5] Creating SQLite indexes in subset database...")
    sub_conn = sqlite3.connect(SUBSET_DB)
    sub_cursor = sub_conn.cursor()
    
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_sub_pat_sub_id ON patients(subject_id)",
        "CREATE INDEX IF NOT EXISTS idx_sub_adm_sub_id ON admissions(subject_id)",
        "CREATE INDEX IF NOT EXISTS idx_sub_adm_hadm_id ON admissions(hadm_id)",
        "CREATE INDEX IF NOT EXISTS idx_sub_diag_sub_id ON diagnoses_icd(subject_id)",
        "CREATE INDEX IF NOT EXISTS idx_sub_diag_hadm_id ON diagnoses_icd(hadm_id)",
        "CREATE INDEX IF NOT EXISTS idx_sub_pres_sub_id ON prescriptions(subject_id)",
        "CREATE INDEX IF NOT EXISTS idx_sub_pres_hadm_id ON prescriptions(hadm_id)",
        "CREATE INDEX IF NOT EXISTS idx_sub_lab_sub_id  ON labevents(subject_id)",
        "CREATE INDEX IF NOT EXISTS idx_sub_lab_hadm_id  ON labevents(hadm_id)",
        "CREATE INDEX IF NOT EXISTS idx_sub_lab_item     ON labevents(itemid)"
    ]
    
    for idx_sql in indexes:
        t0 = time.time()
        sub_cursor.execute(idx_sql)
        print(f"  - Index created: {idx_sql.split()[-1]} in {time.time()-t0:.2f}s")
        
    sub_conn.commit()
    
    # Show counts
    print("\n── Verification of Subset DB Row Counts ──")
    tables = tables_to_copy + ["labevents"]
    for table in tables:
        count = sub_cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:<20} : {count:,} rows")
        
    sub_conn.close()
    
    size_mb = os.path.getsize(SUBSET_DB) / (1024 * 1024)
    print(f"\nSubset database successfully generated: {SUBSET_DB}")
    print(f"Total size: {size_mb:.2f} MB")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    extract()

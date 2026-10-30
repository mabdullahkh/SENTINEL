import os
import sys
import subprocess

# ── Auto-bootstrap: if uvicorn isn't visible, re-exec with the venv interpreter ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(BASE_DIR, "venv", "bin", "python")

def _in_venv():
    try:
        import uvicorn  # noqa: F401
        return True
    except ImportError:
        return False

if not _in_venv() and os.path.exists(VENV_PYTHON):
    print("[Bootstrap] Switching to virtual-environment interpreter …")
    os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)
elif not _in_venv():
    print("ERROR: uvicorn not found and no venv detected at ./venv")
    print("Please run:  python3 -m venv venv && venv/bin/pip install -r requirements.txt")
    sys.exit(1)

# ── Path setup (needed when running as plain script) ──
sys.path.insert(0, BASE_DIR)

from backend import config


def main():
    print("=" * 60)
    print("  CLINICAL SENTINEL EHR DECISION SUPPORT SYSTEM")
    print("=" * 60)

    # 1. Check database
    if not os.path.exists(config.DB_PATH):
        print("\n[DB Check] Database subset not found. Initiating extraction…")
        subprocess.run([sys.executable, os.path.join(BASE_DIR, "backend", "extract_subset.py")], cwd=BASE_DIR, check=True)
    else:
        print("[DB Check] Database subset found.")

    # 2. Check trained models
    models_trained = all(
        os.path.exists(os.path.join(config.MODELS_DIR, f"{disease}_{m_type}.pkl"))
        for disease in config.DISEASES
        for m_type in config.MODEL_TYPES
    )
    if not models_trained:
        print("\n[Model Check] Trained models not found. Running training and evaluation…")
        subprocess.run([sys.executable, "-m", "backend.train_and_eval"], cwd=BASE_DIR, check=True)
    else:
        print("[Model Check] Trained models and thresholds found.")

    # 3. Launch server using the venv Python directly (avoids macOS subprocess resolution)
    print("\n[Server Start] Launching clinician dashboard on http://localhost:8001 …")
    subprocess.run([
        VENV_PYTHON, "-m", "uvicorn",
        "backend.app:app",
        "--host", "127.0.0.1",
        "--port", "8001"
    ], cwd=BASE_DIR, check=True)


if __name__ == "__main__":
    main()

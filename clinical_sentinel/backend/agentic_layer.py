import os
import requests
import json
from . import config

class RAGKnowledgeBase:
    """A lightweight RAG retriever that loads clinical guidelines from text files."""
    def __init__(self):
        self.guidelines = {}
        self._initialize_default_guidelines()

    def _initialize_default_guidelines(self):
        # Create default guidelines in the RAG folder if empty
        os.makedirs(config.RAG_DIR, exist_ok=True)
        
        defaults = {
            "sepsis": """
Surviving Sepsis Campaign Guidelines (2021):
1. Measure lactate level. Re-measure if initial lactate is > 2 mmol/L.
2. Obtain blood cultures prior to administration of antibiotics.
3. Administer broad-spectrum antibiotics within 1 hour of recognition.
4. Administer 30 mL/kg crystalloid fluid for hypotension or lactate >= 4 mmol/L within 3 hours.
5. Apply vasopressors (norepinephrine first-line) if patient is hypotensive during or after fluid resuscitation.
""",
            "cardiac": """
ACC/AHA Heart Failure Guidelines:
1. Stage patient risk (Stage A-D) and classification (HFrEF vs HFpEF).
2. Initiate Guideline-Directed Medical Therapy (GDMT): ARNI/ACEi/ARB, beta-blocker, MRA, and SGLT2i for HFrEF.
3. Manage volume status with loop diuretics (e.g., Furosemide) in patients with fluid retention.
4. For acute coronary syndrome (ACS/MI): administer Aspirin, P2Y12 inhibitor, heparin, and evaluate for urgent reperfusion (PCI/Thrombolysis).
""",
            "respiratory": """
GOLD (COPD) & ATS (Pneumonia) Guidelines:
1. Assess severity using CURB-65 score for community-acquired pneumonia.
2. Initiate empiric antibiotic therapy (e.g., macrolide + beta-lactam) within 4 hours for pneumonia.
3. For acute COPD exacerbations: administer short-acting bronchodilators, systemic corticosteroids (prednisone 40mg for 5 days), and oxygen therapy (target saturation 88-92%).
4. Provide non-invasive ventilation (NIV) for respiratory acidosis (pH < 7.35).
""",
            "renal": """
KDIGO Guidelines for Acute Kidney Injury (AKI):
1. Stage AKI based on serum creatinine increase (Stage 1: 1.5-1.9x baseline; Stage 2: 2.0-2.9x baseline; Stage 3: >=3x baseline or initiation of RRT).
2. Discontinue all nephrotoxic agents (NSAIDs, aminoglycosides, contrast media) when possible.
3. Ensure volume status optimization and monitor hemodynamics.
4. Monitor serum creatinine and urine output hourly in critical care.
""",
            "neurological": """
AHA/ASA Stroke & Intracranial Hemorrhage Guidelines:
1. Perform urgent non-contrast Head CT or Brain MRI to differentiate ischemic stroke from hemorrhage.
2. For acute ischemic stroke: evaluate eligibility for IV alteplase (thrombolysis) within 3 to 4.5 hours of last known well.
3. Control blood pressure: target BP < 180/105 mmHg for thrombolysis; avoid aggressive lowering in acute ischemic stroke unless > 220/120.
4. For intracranial hemorrhage: reverse anticoagulation, manage ICP (mannitol/hypertonic saline), and evaluate for surgical decompression.
"""
        }
        
        for disease, text in defaults.items():
            path = os.path.join(config.RAG_DIR, f"{disease}_guidelines.txt")
            if not os.path.exists(path):
                with open(path, 'w') as f:
                    f.write(text.strip())
            self.guidelines[disease] = text.strip()

    def retrieve(self, query_topic):
        """Retrieve guidelines matching the query topic."""
        topic = str(query_topic).lower()
        for key in self.guidelines:
            if key in topic:
                return self.guidelines[key]
        return "General ICU Monitoring Guidelines: Ensure continuous vital monitoring, regular lab evaluations, and multidisciplinary coordination."


class ClinicalAgentBrain:
    """Coordinates patient consultation agents: Historian, Specialists, and Diagnostician."""
    def __init__(self):
        self.rag = RAGKnowledgeBase()

    def is_ollama_active(self):
        """Check if local Ollama is active and responsive."""
        try:
            res = requests.get(f"{config.OLLAMA_API_URL}/api/tags", timeout=1.5)
            return res.status_code == 200
        except Exception:
            return False

    def query_ollama(self, system_prompt, user_prompt):
        """Send chat query to Ollama LLM."""
        try:
            payload = {
                "model": config.LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False
            }
            res = requests.post(f"{config.OLLAMA_API_URL}/api/chat", json=payload, timeout=8.0)
            if res.status_code == 200:
                data = res.json()
                return data['message']['content']
        except Exception as e:
            print(f"Ollama query failed: {e}")
        return None

    def run_consultation(self, patient_info, ml_probabilities, shap_explanations):
        """Runs end-to-end multi-agent clinical consultation."""
        print(f"\n[Multi-Agent Brain] Beginning consultation for Patient ID {patient_info['subject_id']}...")
        
        # 1. Historian Agent
        historian_sys = "You are a clinical Historian Agent. Your role is to analyze a patient's electronic health records, medications, and laboratory timelines to construct a concise chronological patient history profile."
        historian_user = f"""
Patient Profile:
- Subject ID: {patient_info['subject_id']}
- Gender: {patient_info['gender']}
- Age: {patient_info['anchor_age']}
- Primary Admission Type: {patient_info.get('admission_type', 'Emergency')}
- Active Diagnoses: {', '.join([d['long_title'] for d in patient_info.get('diagnoses', [])[:5]])}
- Active Medications: {', '.join([m['drug'] for m in patient_info.get('prescriptions', [])[:6]])}

Create a structured medical history summary. Highlight demographic risk, primary reasons for admission, and active therapies.
"""
        
        print("  - Running Historian Agent (constructing profile)...")
        historian_opinion = None
        if self.is_ollama_active():
            historian_opinion = self.query_ollama(historian_sys, historian_user)
            
        if not historian_opinion:
            # Rule-based fallback
            meds = [m['drug'] for m in patient_info.get('prescriptions', [])[:4]]
            diags = [d['long_title'] for d in patient_info.get('diagnoses', [])[:3]]
            historian_opinion = f"""
**PATIENT SUMMARY RECORD**
- **Demographics:** {patient_info['anchor_age']}-year-old {patient_info['gender']}.
- **Admission Type:** {patient_info.get('admission_type', 'Emergency')}.
- **Clinical Presentation:** The patient presents with a history of {', '.join(diags) if diags else 'acute clinical deterioration'}.
- **Current Active Pharmacotherapy:** Administered medications include {', '.join(meds) if meds else 'standard supportive therapies'}.
- **EHR Assessment:** Patient is at risk of complications due to underlying comorbid state. Close monitoring of vital signs and laboratory parameters is recommended.
"""

        # 2. Specialist Agents
        specialist_opinions = {}
        for disease in config.DISEASES:
            prob = ml_probabilities.get(disease, 0.0)
            shaps = shap_explanations.get(disease, [])
            
            # Fetch guidelines
            guideline = self.rag.retrieve(disease)
            
            specialist_sys = f"You are a Clinical {disease.capitalize()} Specialist Agent. Analyze patient history and ML risk data, cross-reference clinical guidelines, and provide a specialist assessment."
            specialist_user = f"""
Patient History Summary:
{historian_opinion}

ML Risk Report:
- Probability of {disease.upper()}: {prob*100:.1f}%
- Key Contributing Clinical Features (SHAP): {', '.join([f"{s['feature']} (impact: {s['shap_value']:.3f})" for s in shaps])}

Clinical Reference Guidelines:
{guideline}

Provide:
1. Risk Assessment (Severe / Moderate / Low) based on the {prob*100:.1f}% ML probability.
2. Clinical Rationale linking the contributor features (SHAP) to the guidelines.
3. Immediate Specialist Recommendations.
"""
            
            print(f"  - Running {disease.capitalize()} Specialist Agent...")
            opinion = None
            if self.is_ollama_active():
                opinion = self.query_ollama(specialist_sys, specialist_user)
                
            if not opinion:
                # Rule-based fallback
                risk_level = "LOW"
                if prob >= 0.70: risk_level = "CRITICAL / SEVERE"
                elif prob >= config.DEFAULT_THRESHOLDS[disease]: risk_level = "MODERATE / ELEVATED"
                
                shaps_str = ", ".join([f"{s['feature']}" for s in shaps[:3]])
                opinion = f"""
### {disease.capitalize()} Specialist Assessment
- **Risk Level:** {risk_level} (ML Probability: {prob*100:.1f}%)
- **Diagnostic Rationale:** The machine learning model identifies elevated markers. Key drivers include: {shaps_str}. 
- **Guideline Cross-Reference:** According to guidelines, this presentation warrants active surveillance. { 'Immediate intervention per protocol is recommended.' if risk_level == 'CRITICAL / SEVERE' else 'Routine monitoring of electrolyte and fluid status is advised.'}
- **Action Plan:** { '1. Initiate targeted therapy. 2. Repeat blood tests in 4 hours. 3. Monitor urine output.' if 'renal' in disease or 'sepsis' in disease else '1. Continuous cardiac telemetry. 2. Verify oxygen saturation. 3. Optimize guideline-directed medical therapy (GDMT).' }
"""
            specialist_opinions[disease] = opinion

        # 3. Diagnostician Agent (Agent C)
        diagnostician_sys = "You are a Chief Clinical Diagnostician Agent. Your role is to synthesize reviews from multiple specialists, weigh machine learning predictions, and issue a final clinical diagnosis, confidence score, and a consolidated action plan."
        
        specs_summary = "\n\n".join([f"=== {d.upper()} SPECIALIST ===\n{op}" for d, op in specialist_opinions.items()])
        
        diagnostician_user = f"""
Historian Summary:
{historian_opinion}

Specialist Consultations:
{specs_summary}

ML Predicted Probabilities:
{json.dumps(ml_probabilities, indent=2)}

Please output your response in this EXACT structured JSON format:
{{
  "final_assessment": "Short summary of the overall patient status and key findings...",
  "primary_diagnoses": ["Diagnosis 1", "Diagnosis 2"],
  "confidence_score": 85,
  "action_plan": [
    "Step 1...",
    "Step 2..."
  ]
}}
"""
        
        print("  - Running Diagnostician Agent (synthesizing consultation)...")
        diagnostician_opinion_raw = None
        if self.is_ollama_active():
            diagnostician_opinion_raw = self.query_ollama(diagnostician_sys, diagnostician_user)
            
        diagnostician_output = {}
        if diagnostician_opinion_raw:
            try:
                # Extract JSON block from LLM output if any markdown wrapping is present
                json_start = diagnostician_opinion_raw.find("{")
                json_end = diagnostician_opinion_raw.rfind("}") + 1
                if json_start != -1 and json_end != -1:
                    diagnostician_output = json.loads(diagnostician_opinion_raw[json_start:json_end])
            except Exception as e:
                print(f"Failed to parse Diagnostician output JSON: {e}")
                
        if not diagnostician_output:
            # Rule-based fallback synthesis
            max_prob_disease = max(ml_probabilities, key=ml_probabilities.get)
            max_prob = ml_probabilities[max_prob_disease]
            
            primary_diags = []
            actions = ["Maintain baseline vitals recording and intensive care telemetry."]
            
            if max_prob >= config.DEFAULT_THRESHOLDS[max_prob_disease]:
                primary_diags.append(f"Acute {max_prob_disease.capitalize()} Complication")
                actions.append(f"Activate clinical pathway for {max_prob_disease.upper()}.")
                if max_prob_disease == 'sepsis':
                    actions.append("Order broad-spectrum antibiotic infusion and lactate checks.")
                elif max_prob_disease == 'renal':
                    actions.append("Discontinue nephrotoxic agents and track fluid balances hourly.")
                elif max_prob_disease == 'cardiac':
                    actions.append("Administer loop diuretics and perform ECG telemetry.")
            else:
                primary_diags.append("Hemodynamically Stable Patient")
                actions.append("Routine nursing care and discharge evaluation.")
                
            conf = int(max_prob * 100) if max_prob > 0 else 75
            
            diagnostician_output = {
                "final_assessment": f"Patient is primary risk-loaded for {max_prob_disease.upper()} (probability {max_prob*100:.1f}%). Multi-agent consultation indicates an active, guideline-driven monitoring plan is required.",
                "primary_diagnoses": primary_diags,
                "confidence_score": min(95, max(55, conf)),
                "action_plan": actions
            }

        return {
            "historian": historian_opinion,
            "specialists": specialist_opinions,
            "diagnostician": diagnostician_output
        }

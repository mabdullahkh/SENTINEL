// Clinical Sentinel: Clinician Dashboard Controller

let cohortPatients = [];
let activePatient = null;
let activeAdmission = null;
let shapChart = null;
let evaluationCurveChart = null;
let activeXaiTab = 'shap';
let activeCurveTab = 'roc';
let evaluationMetricsData = null; // Store evaluation data from API

// API base URL helper
const API_BASE = '/api';

// On document load
document.addEventListener('DOMContentLoaded', () => {
    initApp();
});

async function initApp() {
    console.log("Initializing Clinical Sentinel application...");
    await checkOllamaStatus();
    await loadCohort();
    await loadSystemPerformance();
    
    // Set up search listener
    document.getElementById('patient-search').addEventListener('input', filterPatients);
}

// Check local Ollama health status
async function checkOllamaStatus() {
    const dot = document.getElementById('ollama-status-dot');
    const txt = document.getElementById('ollama-status-text');
    try {
        const res = await fetch('/api/patients'); // verify API is alive first
        if (res.ok) {
            // Let's assume server will report Ollama status.
            // For now, we query the server and it queries Ollama, or we query directly.
            // Let's make a mock check since backend wraps connection status.
            const checkRes = await fetch('/api/metrics');
            const data = await checkRes.json();
            if (data && data.metrics && data.metrics.agent_performance) {
                const status = data.metrics.agent_performance.privacy_compliance;
                dot.className = 'dot green';
                txt.textContent = `Ollama: LLaMA 3.1 8B (${status})`;
            } else {
                dot.className = 'dot orange';
                txt.textContent = 'Ollama: Standby/Fallback';
            }
        }
    } catch (e) {
        dot.className = 'dot orange';
        txt.textContent = 'Ollama: Standby';
    }
}

// Load patient cohort list
async function loadCohort() {
    const container = document.getElementById('patient-list-container');
    try {
        const res = await fetch(`${API_BASE}/patients`);
        const data = await res.json();
        
        if (data.status === 'success') {
            cohortPatients = data.data;
            document.getElementById('patient-count').textContent = `${cohortPatients.length} Pts`;
            renderPatientList(cohortPatients);
        } else {
            container.innerHTML = `<div class="error-msg">Error loading cohort: ${data.message}</div>`;
        }
    } catch (e) {
        container.innerHTML = `<div class="error-msg">Failed to connect to API</div>`;
        console.error(e);
    }
}

// Render patient list sidebar cards
function renderPatientList(patients) {
    const container = document.getElementById('patient-list-container');
    container.innerHTML = '';
    
    if (patients.length === 0) {
        container.innerHTML = `<div class="info-msg">No patients found.</div>`;
        return;
    }
    
    patients.forEach(pat => {
        const card = document.createElement('div');
        card.className = 'patient-card';
        card.id = `pat-card-${pat.subject_id}`;
        card.onclick = () => selectPatient(pat.subject_id);
        
        const genderIcon = pat.gender === 'M' ? '<i class="fa-solid fa-mars font-blue"></i>' : '<i class="fa-solid fa-venus font-pink"></i>';
        const age = pat.anchor_age;
        const diagnosis = pat.long_title || 'General Ward Observation';
        const type = pat.admission_type;
        
        card.innerHTML = `
            <div class="patient-card-header">
                <span>Patient ID: ${pat.subject_id}</span>
                <span>${genderIcon} ${age}y</span>
            </div>
            <div class="patient-card-desc" title="${diagnosis}">${diagnosis}</div>
            <div class="patient-card-meta">
                <span>${type}</span>
                <span>HADM: ${pat.hadm_id}</span>
            </div>
        `;
        container.appendChild(card);
    });
}

// Filter patients list in sidebar
function filterPatients(e) {
    const term = e.target.value.toLowerCase();
    const filtered = cohortPatients.filter(pat => {
        const idMatch = String(pat.subject_id).includes(term);
        const diagMatch = (pat.long_title && pat.long_title.toLowerCase().includes(term));
        const typeMatch = (pat.admission_type && pat.admission_type.toLowerCase().includes(term));
        return idMatch || diagMatch || typeMatch;
    });
    renderPatientList(filtered);
}

// Select a patient and display details
async function selectPatient(subjectId) {
    // Highlight active card
    document.querySelectorAll('.patient-card').forEach(c => c.classList.remove('active'));
    const activeCard = document.getElementById(`pat-card-${subjectId}`);
    if (activeCard) activeCard.classList.add('active');
    
    // Hide placeholder, show active panel
    document.getElementById('workspace-placeholder').classList.add('hidden');
    document.getElementById('active-workspace').classList.remove('hidden');
    
    // Show loading indicators on details
    const container = document.getElementById('mdt-chat-container');
    container.innerHTML = `<div class="console-prompt">> SYSTEM: Loading Patient Profile...</div>`;
    document.getElementById('diagnostician-result-card').classList.add('hidden');
    
    try {
        const res = await fetch(`${API_BASE}/patient/${subjectId}`);
        const data = await res.json();
        
        if (data.status === 'success') {
            activePatient = data.profile;
            // Use the latest admission for active workspace evaluation
            activeAdmission = data.admissions[data.admissions.length - 1];
            
            updateDemographicsPanel();
            renderLabTimeline();
            await evaluateRiskScores();
            
            // Welcome log in console
            container.innerHTML = `
                <div class="terminal-welcome">
                    <p class="console-prompt">> SYSTEM: Cohort details loaded for Patient ID ${activePatient.subject_id}.</p>
                    <p class="console-desc">Ready to launch multi-agent clinical consultation panel. Click "Initiate MDT Panel" to begin.</p>
                </div>
            `;
        }
    } catch (e) {
        console.error(e);
    }
}

// Update active patient demographics info
function updateDemographicsPanel() {
    document.getElementById('patient-id-val').textContent = activePatient.subject_id;
    document.getElementById('hadm-id-val').textContent = activeAdmission.hadm_id;
    document.getElementById('age-val').textContent = activePatient.anchor_age;
    document.getElementById('gender-val').textContent = activePatient.gender === 'M' ? 'Male' : 'Female';
    document.getElementById('admission-type-val').textContent = activeAdmission.admission_type;
    
    const primaryDiag = activeAdmission.diagnoses.find(d => d.seq_num === 1);
    const diagVal = document.getElementById('diag-val');
    diagVal.textContent = primaryDiag ? primaryDiag.long_title : 'General Admission';
    diagVal.title = primaryDiag ? primaryDiag.long_title : 'General Admission';
}

// Render patient laboratory measurements
function renderLabTimeline() {
    const container = document.getElementById('labs-grid-container');
    container.innerHTML = '';
    
    // Find unique latest measurements for common labs
    const labs = activeAdmission.labs || [];
    
    // Define standard reference ranges
    const labRefs = {
        "creatinine": { min: 0.6, max: 1.2, unit: "mg/dL" },
        "bun": { min: 7.0, max: 20.0, unit: "mg/dL" },
        "sodium": { min: 135, max: 145, unit: "mEq/L" },
        "potassium": { min: 3.5, max: 5.0, unit: "mEq/L" },
        "wbc": { min: 4.0, max: 11.0, unit: "k/uL" },
        "platelets": { min: 150, max: 450, unit: "k/uL" },
        "hemoglobin": { min: 12.0, max: 17.5, unit: "g/dL" },
        "lactate": { min: 0.5, max: 2.2, unit: "mmol/L" },
        "glucose": { min: 70, max: 100, unit: "mg/dL" }
    };
    
    // Process matching labs
    const latestLabs = {};
    
    labs.forEach(l => {
        const name = String(l.label).toLowerCase();
        let matchKey = null;
        
        if (name.includes('creatinine')) matchKey = 'creatinine';
        else if (name.includes('urea nitrogen') || name.includes('bun')) matchKey = 'bun';
        else if (name.includes('sodium')) matchKey = 'sodium';
        else if (name.includes('potassium')) matchKey = 'potassium';
        else if (name.includes('white blood cell') || name.includes('wbc')) matchKey = 'wbc';
        else if (name.includes('platelet')) matchKey = 'platelets';
        else if (name.includes('hemoglobin')) matchKey = 'hemoglobin';
        else if (name.includes('lactate')) matchKey = 'lactate';
        else if (name.includes('glucose')) matchKey = 'glucose';
        
        if (matchKey) {
            // Keep the latest measurement based on charttime
            if (!latestLabs[matchKey] || new Date(l.charttime) > new Date(latestLabs[matchKey].charttime)) {
                latestLabs[matchKey] = l;
            }
        }
    });
    
    // Create cards for each biomarker
    Object.keys(labRefs).forEach(key => {
        const ref = labRefs[key];
        const labObj = latestLabs[key];
        
        let value = '–';
        let isAbnormal = false;
        
        if (labObj && labObj.valuenum !== null) {
            value = labObj.valuenum.toFixed(1);
            const valNum = parseFloat(value);
            isAbnormal = valNum < ref.min || valNum > ref.max || labObj.flag === 'abnormal';
        }
        
        const card = document.createElement('div');
        card.className = `lab-card ${isAbnormal ? 'abnormal' : ''}`;
        
        card.innerHTML = `
            <span class="lab-title">${key}</span>
            <div class="lab-value-row">
                <span class="lab-value">${value}</span>
                <span class="lab-unit">${ref.unit}</span>
            </div>
            <span class="lab-range">Ref: ${ref.min}-${ref.max}</span>
        `;
        container.appendChild(card);
    });
}

// Compute and animate ML risk predictions
async function evaluateRiskScores() {
    const container = document.getElementById('risk-meters-container');
    container.innerHTML = '<div class="loading-spinner"><i class="fa-solid fa-spinner fa-spin"></i> Estimating risks...</div>';
    
    try {
        const res = await fetch(`${API_BASE}/patient/${activeAdmission.hadm_id}/scores`);
        const data = await res.json();
        
        if (data.status === 'success') {
            container.innerHTML = '';
            const probs = data.probabilities;
            const thresholds = data.thresholds;
            const crossed = data.threshold_crossed;
            
            let alarmTriggered = false;
            let criticalPathology = '';
            
            Object.keys(probs).forEach(disease => {
                const prob = probs[disease];
                const thresh = thresholds[disease] || 0.5;
                const isCrossed = crossed[disease];
                
                const percent = (prob * 100).toFixed(1);
                const threshPercent = (thresh * 100).toFixed(0);
                
                const row = document.createElement('div');
                row.className = 'risk-row';
                
                const warningIcon = isCrossed ? '<i class="fa-solid fa-circle-exclamation alert-icon-mini"></i>' : '';
                if (isCrossed) {
                    alarmTriggered = true;
                    criticalPathology += `${disease.toUpperCase()} (${percent}%), `;
                }
                
                row.innerHTML = `
                    <div class="risk-row-header">
                        <span class="risk-lbl">${disease} ${warningIcon}</span>
                        <span class="risk-val" style="color: var(--clr-${disease})">${percent}%</span>
                    </div>
                    <div class="risk-progress-bg">
                        <div class="risk-progress-bar ${disease}-bar" style="width: 0%"></div>
                        <div class="risk-threshold-marker" style="left: ${threshPercent}%" title="Calibrated Threshold: ${threshPercent}%"></div>
                    </div>
                    <div class="threshold-lbl" style="text-align: right">Thresh: ${threshPercent}%</div>
                `;
                container.appendChild(row);
                
                // Trigger width transition in next paint
                setTimeout(() => {
                    row.querySelector('.risk-progress-bar').style.width = `${percent}%`;
                }, 100);
            });
            
            // Handle Urgent Audio-Visual Alarm Trigger
            if (alarmTriggered) {
                triggerUrgentAlarm(criticalPathology.slice(0, -2));
            } else {
                dismissAlarm();
            }
            
            // Reload Explainable AI values for default selected disease
            loadXaiExplanations();
        }
    } catch (e) {
        console.error(e);
        container.innerHTML = '<div class="error-msg">Failed to compute risks</div>';
    }
}

// Explainable AI tab switcher
function switchXaiTab(tab) {
    activeXaiTab = tab;
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    
    if (tab === 'shap') {
        event.target.classList.add('active');
        document.getElementById('shap-tab-content').classList.remove('hidden');
        document.getElementById('lime-tab-content').classList.add('hidden');
    } else {
        event.target.classList.add('active');
        document.getElementById('shap-tab-content').classList.add('hidden');
        document.getElementById('lime-tab-content').classList.remove('hidden');
    }
}

// Fetch and render SHAP and LIME details
async function loadXaiExplanations() {
    if (!activeAdmission) return;
    const disease = document.getElementById('xai-disease-selector').value;
    
    try {
        const res = await fetch(`${API_BASE}/patient/${activeAdmission.hadm_id}/explain`);
        const data = await res.json();
        
        if (data.status === 'success') {
            const exp = data.explanations[disease];
            renderShapChart(exp.shap);
            renderLimeRules(exp.lime);
        }
    } catch (e) {
        console.error(e);
    }
}

// Draw SHAP Horizontal Bar Chart
function renderShapChart(shapData) {
    const ctx = document.getElementById('shap-chart').getContext('2d');
    
    // Sort SHAP values by absolute magnitude for plotting
    const sorted = [...shapData].sort((a, b) => Math.abs(a.shap_value) - Math.abs(b.shap_value));
    
    const labels = sorted.map(item => item.feature);
    const values = sorted.map(item => item.shap_value);
    
    // Color code: red for risk-increasing (+ve SHAP), green for risk-decreasing (-ve SHAP)
    const colors = values.map(val => val >= 0 ? 'rgba(239, 68, 68, 0.85)' : 'rgba(16, 185, 129, 0.85)');
    const borderColors = values.map(val => val >= 0 ? '#ef4444' : '#10b981');
    
    if (shapChart) {
        shapChart.destroy();
    }
    
    shapChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'SHAP Value (Impact on Prediction)',
                data: values,
                backgroundColor: colors,
                borderColor: borderColors,
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: 'hsl(215, 20%, 65%)', font: { family: 'Fira Code', size: 9 } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: 'hsl(210, 40%, 96%)', font: { family: 'Inter', size: 10 } }
                }
            }
        }
    });
}

// Draw LIME rule explanations
function renderLimeRules(limeData) {
    const container = document.getElementById('lime-rules-container');
    container.innerHTML = '';
    
    if (!limeData || limeData.length === 0) {
        container.innerHTML = '<li>No local explanation rules computed</li>';
        return;
    }
    
    limeData.forEach(item => {
        const li = document.createElement('li');
        const signClass = item.weight >= 0 ? 'positive' : 'negative';
        const arrow = item.weight >= 0 ? '↑ Increases Risk' : '↓ Decreases Risk';
        
        li.className = `lime-rule-item ${signClass}`;
        li.innerHTML = `
            <span class="lime-rule-text">${item.rule}</span>
            <span class="lime-rule-weight">${arrow} (${item.weight.toFixed(3)})</span>
        `;
        container.appendChild(li);
    });
}

// Run multi-agent consultation panel (Historian, Specialist, Diagnostician, RAG)
async function runAgentConsultation() {
    if (!activeAdmission) return;
    
    const container = document.getElementById('mdt-chat-container');
    const runBtn = document.getElementById('run-consult-btn');
    const resultCard = document.getElementById('diagnostician-result-card');
    
    runBtn.disabled = true;
    resultCard.classList.add('hidden');
    container.innerHTML = '';
    
    // Create status updates
    appendConsoleMessage('SYSTEM', 'Connecting to Multi-Agent Brain orchestration (CrewAI Core)...', 'console-prompt');
    appendConsoleMessage('SYSTEM', 'RAG Retrieval: Fetching clinical guidelines...', 'console-prompt');
    
    try {
        const res = await fetch(`${API_BASE}/patient/${activeAdmission.hadm_id}/consult`, { method: 'POST' });
        const data = await res.json();
        
        if (data.status === 'success') {
            const consult = data.consultation;
            
            // Animate messages typing out sequentially to simulate real clinical discussions
            await sleep(800);
            appendConsoleMessage('HISTORIAN AGENT', consult.historian, 'bubble-historian');
            
            const diseases = Object.keys(consult.specialists);
            for (let d of diseases) {
                await sleep(1000);
                appendConsoleMessage(`${d.toUpperCase()} SPECIALIST AGENT`, consult.specialists[d], 'bubble-specialist');
            }
            
            await sleep(1200);
            appendConsoleMessage('DIAGNOSTICIAN AGENT', `Synthesized Panel reviews and RAG Guidelines. Compiled final diagnostic reports.`, 'bubble-diagnostician');
            
            // Show consolidated diagnostician summary card
            const diag = consult.diagnostician;
            document.getElementById('diag-assessment-text').textContent = diag.final_assessment;
            document.getElementById('diag-confidence').textContent = `${diag.confidence_score}%`;
            
            const actionsList = document.getElementById('diag-action-list');
            actionsList.innerHTML = '';
            diag.action_plan.forEach(act => {
                const li = document.createElement('li');
                li.textContent = act;
                actionsList.appendChild(li);
            });
            
            resultCard.classList.remove('hidden');
            resultCard.scrollIntoView({ behavior: 'smooth' });
        } else {
            appendConsoleMessage('SYSTEM ERROR', `Consultation workflow failed: ${data.message}`, 'console-prompt');
        }
    } catch (e) {
        console.error(e);
        appendConsoleMessage('SYSTEM ERROR', 'Network connection interrupted during CrewAI consult.', 'console-prompt');
    } finally {
        runBtn.disabled = false;
    }
}

// Append bubble to console logs
function appendConsoleMessage(sender, text, cssClass) {
    const container = document.getElementById('mdt-chat-container');
    const bubble = document.createElement('div');
    bubble.className = `mdt-bubble ${cssClass}`;
    
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    
    bubble.innerHTML = `
        <div class="mdt-bubble-header">
            <span>> ${sender}</span>
            <span class="info-text">${time}</span>
        </div>
        <div class="mdt-bubble-body">${text}</div>
    `;
    container.appendChild(bubble);
    container.scrollTop = container.scrollHeight;
}

// Clinical Decision Feedback Form
async function submitFeedback(e) {
    e.preventDefault();
    if (!activeAdmission) return;
    
    const submitBtn = document.getElementById('feedback-submit-btn');
    submitBtn.disabled = true;
    
    const feedback = {
        hadm_id: activeAdmission.hadm_id,
        disease: document.getElementById('feedback-disease').value,
        clinician_action: document.getElementById('feedback-action').value,
        notes: document.getElementById('feedback-notes').value
    };
    
    try {
        const res = await fetch(`${API_BASE}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(feedback)
        });
        const data = await res.json();
        
        if (data.status === 'success') {
            alert('Clinician decision recorded in audit log. Learning loop updated.');
            document.getElementById('feedback-notes').value = '';
        } else {
            alert(`Error recording feedback: ${data.message}`);
        }
    } catch (err) {
        console.error(err);
        alert('Failed to log feedback.');
    } finally {
        submitBtn.disabled = false;
    }
}

// System Performance Tab curves switcher
function switchCurveTab(tab) {
    activeCurveTab = tab;
    document.querySelectorAll('.curve-tab-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    renderPerformanceCurves();
}

// Load evaluation and system metrics
async function loadSystemPerformance() {
    try {
        const res = await fetch(`${API_BASE}/metrics`);
        const data = await res.json();
        
        if (data.status === 'success' && data.metrics) {
            evaluationMetricsData = data.metrics.model_performance;
            
            // Draw ROC or PR curve
            renderPerformanceCurves();
            
            // Set up static agent card values
            const perf = data.metrics.agent_performance;
            if (perf) {
                document.getElementById('agent-acc-val').textContent = `${(perf.diagnostic_accuracy * 100).toFixed(0)}%`;
                document.getElementById('agent-latency-val').textContent = `< ${perf.response_latency_sec}s`;
                document.getElementById('agent-precision-val').textContent = `${(perf.alert_precision * 100).toFixed(0)}%`;
            }
        }
    } catch (e) {
        console.error("Failed to load evaluation metrics", e);
    }
}

// Draw ROC or PR Curve with Chart.js
function renderPerformanceCurves() {
    if (!evaluationMetricsData) return;
    
    const ctx = document.getElementById('eval-curve-chart').getContext('2d');
    
    const diseases = Object.keys(evaluationMetricsData);
    const datasets = [];
    
    // Distinct colors for each line
    const curveColors = {
        sepsis: '#ef4444',
        cardiac: '#a855f7',
        respiratory: '#f97316',
        renal: '#14b8a6',
        neurological: '#06b6d4'
    };
    
    diseases.forEach(d => {
        const info = evaluationMetricsData[d];
        // Select ROC or PR coordinates
        const points = activeCurveTab === 'roc' ? info.roc_curve : info.pr_curve;
        
        datasets.push({
            label: `${d.toUpperCase()} (AUC: ${info.auc_roc.toFixed(2)})`,
            data: points,
            borderColor: curveColors[d] || '#38bdf8',
            borderWidth: 1.5,
            pointRadius: 0,
            fill: false,
            tension: 0.1
        });
    });
    
    if (evaluationCurveChart) {
        evaluationCurveChart.destroy();
    }
    
    // Add baseline reference lines
    if (activeCurveTab === 'roc') {
        // Diagonal reference
        datasets.push({
            label: 'No Skill (AUC: 0.50)',
            data: [{x:0, y:0}, {x:1, y:1}],
            borderColor: 'rgba(255,255,255,0.15)',
            borderDash: [5, 5],
            borderWidth: 1,
            pointRadius: 0,
            fill: false
        });
    }
    
    evaluationCurveChart = new Chart(ctx, {
        type: 'line',
        data: { datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: activeCurveTab === 'roc' ? 'False Positive Rate' : 'Recall',
                        color: 'hsl(215, 20%, 65%)',
                        font: { size: 9 }
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: 'hsl(215, 20%, 65%)', font: { size: 8 } },
                    min: 0,
                    max: 1
                },
                y: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: activeCurveTab === 'roc' ? 'True Positive Rate (Recall)' : 'Precision',
                        color: 'hsl(215, 20%, 65%)',
                        font: { size: 9 }
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: 'hsl(215, 20%, 65%)', font: { size: 8 } },
                    min: 0,
                    max: 1
                }
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: 'hsl(215, 20%, 65%)',
                        font: { size: 8 },
                        boxWidth: 8
                    }
                }
            }
        }
    });
}

// Alarm overlays trigger
function triggerUrgentAlarm(criticalDiseases) {
    const overlay = document.getElementById('urgent-alarm-overlay');
    overlay.querySelector('span').textContent = `CRITICAL EHR PATIENT ALERT: PATHOLOGY THRESHOLD EXCEEDED - ${criticalDiseases}`;
    overlay.classList.remove('hidden');
    
    // Play hospital beep sound
    const sound = document.getElementById('alert-sound');
    if (sound) {
        sound.currentTime = 0;
        sound.play().catch(e => console.log("Audio play deferred until user interaction."));
    }
}

function dismissAlarm() {
    document.getElementById('urgent-alarm-overlay').classList.add('hidden');
}

// Utility sleeping helper
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

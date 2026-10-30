document.addEventListener("DOMContentLoaded", () => {
    const searchInput = document.getElementById("input-search-subject");
    const dropdownSelect = document.getElementById("select-patient-dropdown");
    const loadBtn = document.getElementById("btn-load-patient");
    const patientContent = document.getElementById("patient-content");

    const patientTitle = document.getElementById("patient-title");
    const patientMeta = document.getElementById("patient-meta");
    const openDashBtn = document.getElementById("btn-open-in-dash");
    const visitTabsBar = document.getElementById("visit-tabs-bar");

    const visitHeaderTitle = document.getElementById("visit-header-title");
    const visitSummaryInfo = document.getElementById("visit-summary-info");
    const visitDiagnosesList = document.getElementById("visit-diagnoses-list");
    const visitLabsBody = document.getElementById("visit-labs-table-body");
    const visitRiskCards = document.getElementById("visit-risk-cards");

    const formAddReport = document.getElementById("form-add-report");
    const selectTargetHadm = document.getElementById("select-target-hadm");
    const newVisitFields = document.getElementById("new-visit-fields");
    const submitReportBtn = document.getElementById("btn-submit-report");

    let currentPatientData = null;
    let selectedAdmissionIndex = 0;

    // 1. Fetch patients list for dropdown
    fetchPatientsList();

    async function fetchPatientsList() {
        try {
            const res = await fetch("/api/patients");
            const data = await res.json();
            if (data.status === "success" && data.data) {
                dropdownSelect.innerHTML = `<option value="">-- Select Patient from Cohort (${data.data.length}) --</option>`;

                // Group unique patients
                const seen = new Set();
                data.data.forEach(p => {
                    if (!seen.has(p.subject_id)) {
                        seen.add(p.subject_id);
                        const opt = document.createElement("option");
                        opt.value = p.subject_id;
                        opt.textContent = `Patient #${p.subject_id} (${p.gender}, ${p.anchor_age}y) - ${p.long_title || 'General'}`;
                        dropdownSelect.appendChild(opt);
                    }
                });
            }
        } catch (e) {
            console.error("Failed to load patients dropdown:", e);
        }
    }

    // Handle search or dropdown selection
    loadBtn.addEventListener("click", () => {
        const val = searchInput.value.trim() || dropdownSelect.value;
        if (val) {
            loadPatientHistory(val);
        } else {
            alert("Please enter or select a Patient Subject ID.");
        }
    });

    dropdownSelect.addEventListener("change", () => {
        if (dropdownSelect.value) {
            searchInput.value = dropdownSelect.value;
            loadPatientHistory(dropdownSelect.value);
        }
    });

    // Check URL params for ?subject_id=XYZ or ?hadm_id=XYZ
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.has("subject_id")) {
        loadPatientHistory(urlParams.get("subject_id"));
    }

    // 2. Load Patient History
    async function loadPatientHistory(subjectId) {
        loadBtn.disabled = true;
        loadBtn.innerHTML = `<i class="fa-solid fa-spinner spinner"></i> Loading...`;

        try {
            const res = await fetch(`/api/patient/${subjectId}/history`);
            const data = await res.json();

            if (data.status === "success") {
                currentPatientData = data;
                selectedAdmissionIndex = data.admissions.length - 1; // Default to latest visit
                renderPatientView();
            } else {
                alert("Error: " + (data.message || "Patient history not found"));
            }
        } catch (e) {
            console.error("Failed to load history:", e);
            alert("Failed to connect to backend server.");
        } finally {
            loadBtn.disabled = false;
            loadBtn.innerHTML = `<i class="fa-solid fa-user-check"></i> Inspect History`;
        }
    }

    // 3. Render Patient View & Timeline
    function renderPatientView() {
        patientContent.classList.remove("hidden");
        const profile = currentPatientData.profile;
        const admissions = currentPatientData.admissions;

        patientTitle.textContent = `Patient #${profile.subject_id}`;
        patientMeta.innerHTML = `
            <span><i class="fa-solid fa-user"></i> Gender: <strong>${profile.gender === 'F' ? 'Female' : 'Male'}</strong></span>
            <span><i class="fa-solid fa-cake-candles"></i> Age: <strong>${profile.anchor_age}</strong></span>
            <span><i class="fa-solid fa-hospital"></i> Total Visits: <strong>${admissions.length}</strong></span>
        `;

        // Populate visit tabs
        visitTabsBar.innerHTML = "";
        selectTargetHadm.innerHTML = `<option value="new">+ Create New Hospital Visit</option>`;

        admissions.forEach((adm, idx) => {
            // Tab button
            const tab = document.createElement("button");
            tab.className = `visit-tab ${idx === selectedAdmissionIndex ? 'active' : ''}`;
            const admTimeStr = adm.admittime ? adm.admittime.split(" ")[0] : `Visit ${idx + 1}`;
            tab.innerHTML = `<i class="fa-solid fa-calendar-day"></i> Visit ${idx + 1} (${admTimeStr})`;
            tab.addEventListener("click", () => {
                selectedAdmissionIndex = idx;
                renderPatientView();
            });
            visitTabsBar.appendChild(tab);

            // Target Hadm Dropdown options for Add Report form
            const opt = document.createElement("option");
            opt.value = adm.hadm_id;
            opt.textContent = `Append to Visit ${idx + 1} (Admission #${adm.hadm_id})`;
            selectTargetHadm.appendChild(opt);
        });

        // Render selected visit details
        renderSelectedAdmission(admissions[selectedAdmissionIndex]);
    }

    function renderSelectedAdmission(adm) {
        if (!adm) return;

        openDashBtn.href = `/?hadm_id=${adm.hadm_id}`;

        visitHeaderTitle.innerHTML = `<i class="fa-solid fa-hospital-user"></i> Hospital Visit #${adm.hadm_id}`;
        visitSummaryInfo.innerHTML = `
            <div><strong>Admission Time:</strong> ${adm.admittime || 'N/A'}</div>
            <div><strong>Admission Type:</strong> ${adm.admission_type || 'N/A'} &nbsp;|&nbsp; <strong>Insurance:</strong> ${adm.insurance || 'N/A'}</div>
        `;

        // Diagnoses
        visitDiagnosesList.innerHTML = "";
        if (adm.diagnoses && adm.diagnoses.length > 0) {
            adm.diagnoses.forEach(d => {
                const item = document.createElement("div");
                item.style.cssText = "background: #0f172a; padding: 0.5rem 0.75rem; border-radius: 6px; margin-bottom: 0.35rem; font-size: 0.85rem; border: 1px solid #334155;";
                item.innerHTML = `<strong style="color: #38bdf8;">[${d.icd_code}]</strong> ${d.long_title}`;
                visitDiagnosesList.appendChild(item);
            });
        } else {
            visitDiagnosesList.innerHTML = `<div style="color: #64748b; font-size: 0.85rem;">No primary ICD diagnoses recorded.</div>`;
        }

        // Labs Table
        visitLabsBody.innerHTML = "";
        if (adm.labs && adm.labs.length > 0) {
            adm.labs.forEach(lab => {
                const tr = document.createElement("tr");

                let statusBadge = `<span class="tag-flag tag-normal">NORMAL</span>`;
                if (lab.flag && lab.flag.toLowerCase() === 'abnormal') {
                    statusBadge = `<span class="tag-flag tag-abnormal">ABNORMAL</span>`;
                }

                const refRange = (lab.ref_range_lower !== null && lab.ref_range_upper !== null)
                    ? `${lab.ref_range_lower} - ${lab.ref_range_upper}`
                    : 'N/A';

                tr.innerHTML = `
                    <td><strong>${lab.label || 'Lab #' + lab.itemid}</strong></td>
                    <td style="color: #f8fafc; font-weight: 600;">${lab.valuenum !== null ? lab.valuenum : 'N/A'}</td>
                    <td style="color: #94a3b8;">${lab.valueuom || ''}</td>
                    <td style="color: #64748b;">${refRange}</td>
                    <td>${statusBadge}</td>
                `;
                visitLabsBody.appendChild(tr);
            });
        } else {
            visitLabsBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: #64748b; padding: 1.5rem;">No laboratory test measurements logged for this visit.</td></tr>`;
        }

        // Computed Predictions
        visitRiskCards.innerHTML = "";
        if (adm.predictions && adm.predictions.probabilities) {
            const probs = adm.predictions.probabilities;
            const crossed = adm.predictions.threshold_crossed;

            Object.keys(probs).forEach(disease => {
                const prob = probs[disease];
                const pct = (prob * 100).toFixed(1);
                const isHigh = crossed[disease];

                const card = document.createElement("div");
                card.style.cssText = "background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.65rem;";
                card.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                        <span style="font-weight: 600; text-transform: capitalize; color: #e2e8f0; font-size: 0.85rem;">${disease}</span>
                        <span class="tag-flag ${isHigh ? 'tag-abnormal' : 'tag-normal'}">${isHigh ? 'HIGH RISK' : 'LOW RISK'}</span>
                    </div>
                    <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.35rem;">Risk Score: <strong style="color: #f8fafc;">${pct}%</strong></div>
                    <div style="background: #1e293b; height: 6px; border-radius: 4px; overflow: hidden;">
                        <div style="width: ${pct}%; height: 100%; background: ${isHigh ? '#ef4444' : '#10b981'};"></div>
                    </div>
                `;
                visitRiskCards.appendChild(card);
            });
        } else {
            visitRiskCards.innerHTML = `<div style="color: #64748b; font-size: 0.85rem; padding: 1rem 0;">Risk scoring requires recorded lab biomarker data.</div>`;
        }
    }

    // Toggle new visit fields
    selectTargetHadm.addEventListener("change", () => {
        if (selectTargetHadm.value === "new") {
            newVisitFields.classList.remove("hidden");
        } else {
            newVisitFields.classList.add("hidden");
        }
    });

    // 4. Submit New Report / Visit for Existing Patient
    formAddReport.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (!currentPatientData) return;

        const subjectId = currentPatientData.profile.subject_id;
        const targetHadm = selectTargetHadm.value;

        submitReportBtn.disabled = true;
        submitReportBtn.innerHTML = `<i class="fa-solid fa-spinner spinner"></i> Saving Report...`;

        const getNum = id => {
            const val = document.getElementById(id).value;
            return val !== "" ? parseFloat(val) : null;
        };

        const payload = {
            hadm_id: targetHadm,
            admission_type: document.getElementById("new_admission_type").value,
            primary_diagnosis: document.getElementById("new_primary_diagnosis").value.trim(),
            labs: {
                creatinine: getNum("rep_creatinine"),
                bun: getNum("rep_bun"),
                lactate: getNum("rep_lactate"),
                wbc: getNum("rep_wbc"),
                platelets: getNum("rep_platelets"),
                hemoglobin: getNum("rep_hemoglobin")
            }
        };

        try {
            const res = await fetch(`/api/patient/${subjectId}/add-report`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            if (data.status === "success") {
                alert(data.message);
                // Reload patient history to show updated records
                await loadPatientHistory(subjectId);
            } else {
                alert("Error: " + (data.message || "Failed to add report"));
            }
        } catch (e) {
            console.error("Failed to submit report:", e);
            alert("Failed to submit report. Please check server.");
        } finally {
            submitReportBtn.disabled = false;
            submitReportBtn.innerHTML = `<i class="fa-solid fa-plus-circle"></i> Save New Report & Recalculate`;
        }
    });
});

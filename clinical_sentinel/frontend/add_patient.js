document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("patient-form");
    const submitBtn = document.getElementById("btn-submit");
    const resultsPlaceholder = document.getElementById("results-placeholder");
    const resultsContent = document.getElementById("results-content");
    const recordIdsSpan = document.getElementById("record-ids");
    const riskContainer = document.getElementById("risk-cards-container");
    const openDashBtn = document.getElementById("btn-open-dashboard");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        // UI Loading state
        submitBtn.disabled = true;
        submitBtn.innerHTML = `<i class="fa-solid fa-spinner spinner"></i> Persisting to Database & Scoring...`;

        const payload = {
            gender: document.getElementById("gender").value,
            anchor_age: parseInt(document.getElementById("anchor_age").value, 10),
            admission_type: document.getElementById("admission_type").value,
            insurance: document.getElementById("insurance").value,
            primary_diagnosis: document.getElementById("primary_diagnosis").value.trim(),
            labs: {
                creatinine: parseFloat(document.getElementById("creatinine").value),
                bun: parseFloat(document.getElementById("bun").value),
                lactate: parseFloat(document.getElementById("lactate").value),
                wbc: parseFloat(document.getElementById("wbc").value),
                platelets: parseFloat(document.getElementById("platelets").value),
                hemoglobin: parseFloat(document.getElementById("hemoglobin").value),
                sodium: parseFloat(document.getElementById("sodium").value),
                potassium: parseFloat(document.getElementById("potassium").value),
                glucose: parseFloat(document.getElementById("glucose").value)
            }
        };

        try {
            const response = await fetch("/api/patient/add", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (data.status === "success") {
                renderResults(data);
            } else {
                alert("Error saving record: " + (data.message || "Unknown error"));
            }
        } catch (err) {
            console.error("Submission failed:", err);
            alert("Failed to connect to the server. Please check backend status.");
        } finally {
            submitBtn.disabled = false;
            submitBtn.innerHTML = `<i class="fa-solid fa-database"></i> Save Record & Run Risk Analysis`;
        }
    });

    function renderResults(data) {
        resultsPlaceholder.classList.add("hidden");
        resultsContent.classList.remove("hidden");

        recordIdsSpan.textContent = `Record Saved! Subject ID: ${data.subject_id} | Admission ID: ${data.hadm_id}`;
        openDashBtn.href = `/?hadm_id=${data.hadm_id}`;

        const probs = data.predictions.probabilities;
        const thresholds = data.predictions.thresholds;
        const crossed = data.predictions.threshold_crossed;

        riskContainer.innerHTML = "";

        Object.keys(probs).forEach((disease) => {
            const prob = probs[disease];
            const pct = (prob * 100).toFixed(1);
            const thresh = thresholds[disease] || 0.5;
            const threshPct = (thresh * 100).toFixed(0);
            const isHigh = crossed[disease];

            const card = document.createElement("div");
            card.className = "risk-card";

            card.innerHTML = `
                <div class="risk-header">
                    <span class="disease-name">${disease}</span>
                    <span class="badge-risk ${isHigh ? 'badge-high' : 'badge-low'}">
                        ${isHigh ? 'HIGH RISK' : 'LOW RISK'}
                    </span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #94a3b8;">
                    <span>Risk Probability: <strong style="color: #f8fafc;">${pct}%</strong></span>
                    <span>Tuned Thresh: ${threshPct}%</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill ${isHigh ? 'fill-high' : 'fill-low'}" style="width: ${pct}%;"></div>
                </div>
            `;

            riskContainer.appendChild(card);
        });
    }
});

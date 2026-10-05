/**
 * NeuroScan AI - Dedicated Frontend JavaScript
 * Handles drag-and-drop, multi-image preview, REST API communication,
 * and dynamic rendering of individual & averaged results.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const previewContainer = document.getElementById("previewContainer");
    const previewGrid = document.getElementById("previewGrid");
    const previewCount = document.getElementById("previewCount");
    const clearBtn = document.getElementById("clearBtn");
    const analyzeBtn = document.getElementById("analyzeBtn");

    const emptyState = document.getElementById("emptyState");
    const loadingState = document.getElementById("loadingState");
    const resultsContent = document.getElementById("resultsContent");
    const resultsBadge = document.getElementById("resultsBadge");

    // Consensus Banner Elements
    const consensusBanner = document.getElementById("consensusBanner");
    const consensusTitle = document.getElementById("consensusTitle");
    const confidencePill = document.getElementById("confidencePill");
    const riskSubtitle = document.getElementById("riskSubtitle");
    const riskDesc = document.getElementById("riskDesc");

    // Metrics Row Elements
    const metricTotal = document.getElementById("metricTotal");
    const metricReversals = document.getElementById("metricReversals");
    const metricCorrected = document.getElementById("metricCorrected");
    const metricNormal = document.getElementById("metricNormal");

    // Probability Bars
    const barReversal = document.getElementById("barReversal");
    const barCorrected = document.getElementById("barCorrected");
    const barNormal = document.getElementById("barNormal");
    const probValReversal = document.getElementById("probValReversal");
    const probValCorrected = document.getElementById("probValCorrected");
    const probValNormal = document.getElementById("probValNormal");

    // Models
    const cnnPred = document.getElementById("cnnPred");
    const cnnConf = document.getElementById("cnnConf");
    const svmPred = document.getElementById("svmPred");
    const svmConf = document.getElementById("svmConf");
    const rfPred = document.getElementById("rfPred");
    const rfConf = document.getElementById("rfConf");

    // Individual Gallery & Actions
    const individualSection = document.getElementById("individualSection");
    const sampleGrid = document.getElementById("sampleGrid");
    const downloadReportBtn = document.getElementById("downloadReportBtn");
    const downloadCsvBtn = document.getElementById("downloadCsvBtn");

    // State Variables
    let selectedFiles = [];
    let currentResult = null;

    // Color definitions
    const COLOR_MAP = {
        "Reversal": { bg: "#dc3545", color: "#ffffff", chip: "#ef4444" },
        "Corrected": { bg: "#d97706", color: "#ffffff", chip: "#f59e0b" },
        "Normal": { bg: "#16a34a", color: "#ffffff", chip: "#10b981" }
    };

    // ==========================================
    // File Input & Drag and Drop Handling
    // ==========================================
    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFiles(Array.from(e.dataTransfer.files));
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFiles(Array.from(e.target.files));
        }
    });

    clearBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        resetFiles();
    });

    function resetFiles() {
        selectedFiles = [];
        fileInput.value = "";
        previewGrid.innerHTML = "";
        previewContainer.style.display = "none";
        analyzeBtn.disabled = true;
    }

    function handleFiles(files) {
        const validImages = files.filter(f => f.type.startsWith("image/"));
        if (validImages.length === 0) {
            alert("Please select valid image files (PNG, JPG, JPEG).");
            return;
        }

        selectedFiles = validImages;
        previewGrid.innerHTML = "";
        previewContainer.style.display = "block";
        previewCount.textContent = `${selectedFiles.length} image${selectedFiles.length > 1 ? "s" : ""} selected`;
        analyzeBtn.disabled = false;

        // Render thumbnails
        selectedFiles.forEach((file) => {
            const reader = new FileReader();
            reader.onload = (ev) => {
                const wrap = document.createElement("div");
                wrap.className = "preview-thumb-wrap";
                const img = document.createElement("img");
                img.className = "preview-thumb";
                img.src = ev.target.result;
                img.title = file.name;
                wrap.appendChild(img);
                previewGrid.appendChild(wrap);
            };
            reader.readAsDataURL(file);
        });
    }

    // ==========================================
    // Quick Demo Samples Handling
    // ==========================================
    document.querySelectorAll(".btn-demo").forEach(btn => {
        btn.addEventListener("click", async () => {
            const demoType = btn.getAttribute("data-type");
            triggerLoading(true, "Fetching benchmark sample & computing consensus...");

            try {
                const response = await fetch(`/api/demo?type=${demoType}`);
                const data = await response.json();
                if (data.error) {
                    throw new Error(data.error);
                }
                renderResults(data);
            } catch (err) {
                alert("Error loading demo sample: " + err.message);
                triggerLoading(false);
            }
        });
    });

    // ==========================================
    // Main Analysis Execution
    // ==========================================
    analyzeBtn.addEventListener("click", async () => {
        if (selectedFiles.length === 0) return;

        const formData = new FormData();
        selectedFiles.forEach(f => formData.append("images", f));

        triggerLoading(true, `Analyzing ${selectedFiles.length} handwriting image${selectedFiles.length > 1 ? "s" : ""} & averaging results...`);

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();
            if (data.error) {
                throw new Error(data.error);
            }

            renderResults(data);
        } catch (err) {
            alert("Analysis failed: " + err.message);
            triggerLoading(false);
        }
    });

    function triggerLoading(isLoading, text = "") {
        if (isLoading) {
            emptyState.style.display = "none";
            resultsContent.style.display = "none";
            loadingState.style.display = "block";
            document.getElementById("loadingText").textContent = text;
            resultsBadge.textContent = "Processing...";
            analyzeBtn.disabled = true;
        } else {
            loadingState.style.display = "none";
            analyzeBtn.disabled = selectedFiles.length === 0;
        }
    }

    // ==========================================
    // Dynamic Results Rendering
    // ==========================================
    function renderResults(res) {
        currentResult = res;
        triggerLoading(false);

        emptyState.style.display = "none";
        resultsContent.style.display = "block";
        resultsBadge.textContent = "Analysis Complete";

        const isMulti = res.total_images > 1;
        const predClass = res.overall_prediction;
        const colors = COLOR_MAP[predClass] || COLOR_MAP["Normal"];

        // 1. Executive Banner
        consensusBanner.style.backgroundColor = colors.bg;
        consensusTitle.textContent = predClass.toUpperCase();
        confidencePill.textContent = `Avg Confidence: ${res.overall_confidence_percentage}`;
        riskSubtitle.textContent = `Session Risk Level: ${res.overall_risk_level}`;
        riskDesc.textContent = res.overall_risk_description;

        // 2. Metrics Row
        metricTotal.textContent = res.total_images;
        metricReversals.textContent = `${res.reversal_rate.toFixed(1)}%`;
        metricCorrected.textContent = `${res.corrected_rate.toFixed(1)}%`;
        metricNormal.textContent = `${res.normal_rate.toFixed(1)}%`;

        // 3. Probability Bars (Averaged Across All Samples)
        const pRev = (res.average_probabilities["Reversal"] * 100).toFixed(1);
        const pCorr = (res.average_probabilities["Corrected"] * 100).toFixed(1);
        const pNorm = (res.average_probabilities["Normal"] * 100).toFixed(1);

        barReversal.style.width = `${pRev}%`;
        probValReversal.textContent = `${pRev}%`;

        barCorrected.style.width = `${pCorr}%`;
        probValCorrected.textContent = `${pCorr}%`;

        barNormal.style.width = `${pNorm}%`;
        probValNormal.textContent = `${pNorm}%`;

        // 4. Model Averages Breakdown
        cnnPred.textContent = getTopClass(res.model_averages["CNN"]);
        cnnConf.textContent = `Confidence: ${(res.model_averages["CNN"][cnnPred.textContent] * 100).toFixed(1)}%`;

        svmPred.textContent = getTopClass(res.model_averages["SVM"]);
        svmConf.textContent = `Confidence: ${(res.model_averages["SVM"][svmPred.textContent] * 100).toFixed(1)}%`;

        rfPred.textContent = getTopClass(res.model_averages["Random_Forest"]);
        rfConf.textContent = `Confidence: ${(res.model_averages["Random_Forest"][rfPred.textContent] * 100).toFixed(1)}%`;

        // 5. Individual Samples Gallery
        if (isMulti) {
            individualSection.style.display = "block";
            sampleGrid.innerHTML = "";

            res.individual_results.forEach((sample, idx) => {
                const card = document.createElement("div");
                card.className = "sample-card";

                const thumb = document.createElement("img");
                thumb.className = "sample-thumb";
                thumb.src = sample.processed_thumbnail_base64 || sample.thumbnail_base64 || "/static/images/placeholder.png";
                thumb.title = sample.preprocessing && sample.preprocessing.auto_inverted ? "Model input: Auto-Inverted & Centered" : "Input sample";

                const nameDiv = document.createElement("div");
                nameDiv.className = "sample-name";
                nameDiv.textContent = sample.filename || `Sample ${idx + 1}`;

                const chip = document.createElement("span");
                chip.className = "sample-chip";
                const chipColor = (COLOR_MAP[sample.prediction] || COLOR_MAP["Normal"]).chip;
                chip.style.backgroundColor = chipColor;
                chip.textContent = `${sample.prediction} (${sample.confidence_percentage})`;

                card.appendChild(thumb);
                card.appendChild(nameDiv);
                card.appendChild(chip);

                if (sample.preprocessing && sample.preprocessing.auto_inverted) {
                    const tag = document.createElement("div");
                    tag.style.fontSize = "0.72rem";
                    tag.style.color = "#0284c7";
                    tag.style.marginTop = "3px";
                    tag.textContent = "✨ Auto-Normalized";
                    card.appendChild(tag);
                }

                sampleGrid.appendChild(card);
            });
        } else {
            individualSection.style.display = "none";
        }
    }

    function getTopClass(probDict) {
        let maxVal = -1;
        let maxClass = "Normal";
        for (const [k, v] of Object.entries(probDict)) {
            if (v > maxVal) {
                maxVal = v;
                maxClass = k;
            }
        }
        return maxClass;
    }

    // ==========================================
    // Report & CSV Downloads
    // ==========================================
    downloadReportBtn.addEventListener("click", () => {
        if (!currentResult) return;
        const reportText = generateReportText(currentResult);
        downloadFile(reportText, `Dyslexia_Clinical_Report_${Date.now()}.txt`, "text/plain");
    });

    downloadCsvBtn.addEventListener("click", () => {
        if (!currentResult) return;
        const csvContent = generateCsvContent(currentResult);
        downloadFile(csvContent, `Dyslexia_Screening_Data_${Date.now()}.csv`, "text/csv");
    });

    function generateReportText(res) {
        const indLines = res.individual_results.map((r, i) =>
            `  ${i + 1}. ${r.filename || 'Sample ' + (i+1)} -> ${r.prediction} (${r.confidence_percentage}) [Agreement: ${r.unanimous_agreement ? 'Unanimous' : 'Split'}]`
        ).join("\n");

        return `================================================================================
          NEUROSCAN AI: DYSLEXIA HANDWRITING SCREENING REPORT
================================================================================
Generated Date       : ${new Date().toLocaleString()}
Total Samples Screened: ${res.total_images}
Consensus Decision   : ${res.overall_prediction.toUpperCase()}
Average Confidence   : ${res.overall_confidence_percentage}
Session Risk Level   : ${res.overall_risk_level}
Reversal Frequency   : ${res.reversal_count} / ${res.total_images} samples (${res.reversal_rate.toFixed(1)}%)
Correction Frequency : ${res.corrected_count} / ${res.total_images} samples (${res.corrected_rate.toFixed(1)}%)
Normal Frequency     : ${res.normal_count} / ${res.total_images} samples (${res.normal_rate.toFixed(1)}%)

CLINICAL INTERPRETATION:
${res.overall_risk_description}

AVERAGED PROBABILITY DISTRIBUTION (ALL SAMPLES):
- Corrected : ${(res.average_probabilities["Corrected"] * 100).toFixed(2)}%
- Normal    : ${(res.average_probabilities["Normal"] * 100).toFixed(2)}%
- Reversal  : ${(res.average_probabilities["Reversal"] * 100).toFixed(2)}%

INDIVIDUAL SAMPLE AUDIT TRAIL:
${indLines}

RECOMMENDATION:
${res.reversal_rate >= 20.0 || res.overall_prediction === "Reversal"
    ? "High priority referral for formal psycho-educational diagnostic assessment due to persistent mirror reversal patterns."
    : res.corrected_rate >= 30.0
        ? "Continued developmental monitoring for handwriting motor coordination and letter hesitation."
        : "Standard developmental milestones observed; no clinical dyslexia indication."}

================================================================================
DISCLAIMER: This screening tool provides automated assistive handwriting analysis
and does not replace formal comprehensive clinical diagnosis.
================================================================================
`;
    }

    function generateCsvContent(res) {
        const headers = ["Sample_ID", "Filename", "Consensus_Prediction", "Confidence", "CNN_Pred", "SVM_Pred", "RF_Pred", "Unanimous"];
        const rows = res.individual_results.map((r, i) => [
            i + 1,
            `"${r.filename || 'Sample_' + (i+1)}"`,
            r.prediction,
            r.confidence_percentage,
            r.models.CNN.prediction,
            r.models.SVM.prediction,
            r.models.Random_Forest.prediction,
            r.unanimous_agreement ? "Yes" : "No"
        ].join(","));

        return [headers.join(","), ...rows].join("\n");
    }

    function downloadFile(content, fileName, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = fileName;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
});

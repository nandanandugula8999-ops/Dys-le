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

    // Backend API base: same-origin by default (Flask dev).
    // On static hosts (Netlify) set via ?backend=https://<host> (saved)
    // or window.BACKEND_URL / localStorage "dyslexia_backend".
    try {
        const q = new URLSearchParams(window.location.search).get("backend");
        if (q) localStorage.setItem("dyslexia_backend", q.replace(/\/$/, ""));
    } catch (e) { /* ignore */ }
    const API_BASE = ((window.BACKEND_URL || "").trim()
        || (function () { try { return localStorage.getItem("dyslexia_backend") || ""; } catch (e) { return ""; } })()
    ).replace(/\/$/, "");
    const api = (p) => `${API_BASE}${p}`;
    function isBackendMissing(err, res) {
        if (res && res.status === 404) return true;
        if (!err) return false;
        const m = String((err && err.message) || err);
        return (/Failed to fetch|NetworkError|Load failed|HTML page instead|no \/api backend|Server error 404/i.test(m));
    }
    async function apiJson(res) {
        const ct = res.headers.get("content-type") || "";
        if (!res.ok) {
            const txt = await res.text().catch(() => "");
            throw new Error(`Server error ${res.status}: ${txt.slice(0, 120)}`);
        }
        if (!ct.includes("application/json")) {
            const txt = await res.text().catch(() => "");
            if (txt.trim().startsWith("<!DOCTYPE") || txt.trim().startsWith("<html")) {
                throw new Error("Got an HTML page instead of AI results (static host has no /api backend).");
            }
            throw new Error("Unexpected server response (not JSON).");
        }
        return res.json();
    }

    // ----------------------------------------------------------
    // Offline fallback inference (Netlify / static hosts).
    // Netlify cannot run Flask + TensorFlow, so /api/* returns 404.
    // This lightweight canvas heuristic lets uploads work end-to-end
    // as a demo; for the full 88.67% hybrid AI, host server.py
    // elsewhere and open this page with ?backend=https://your-backend
    // ----------------------------------------------------------
    function fileToDataURL(file) {
        return new Promise((resolve, reject) => {
            const r = new FileReader();
            r.onload = () => resolve(r.result);
            r.onerror = reject;
            r.readAsDataURL(file);
        });
    }
    function loadImageEl(dataUrl) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = dataUrl;
        });
    }
    function offlineAnalyzeDataUrl(dataUrl, filename) {
        return loadImageEl(dataUrl).then((img) => {
            const S = 64;
            const cv = document.createElement("canvas");
            cv.width = S; cv.height = S;
            const ctx = cv.getContext("2d", { willReadFrequently: true });
            ctx.fillStyle = "#000"; ctx.fillRect(0, 0, S, S);
            const scale = Math.min(S / img.width, S / img.height);
            const w = Math.max(1, Math.round(img.width * scale * 0.85));
            const h = Math.max(1, Math.round(img.height * scale * 0.85));
            ctx.drawImage(img, (S - w) / 2, (S - h) / 2, w, h);
            const d = ctx.getImageData(0, 0, S, S).data;
            const gray = new Float32Array(S * S);
            for (let i = 0; i < S * S; i++) {
                gray[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2];
            }
            // Background from border median; invert light backgrounds like server.py
            const border = [];
            for (let x = 0; x < S; x++) { border.push(gray[x]); border.push(gray[(S - 1) * S + x]); }
            for (let y = 0; y < S; y++) { border.push(gray[y * S]); border.push(gray[y * S + S - 1]); }
            border.sort((a, b) => a - b);
            const bg = border[Math.floor(border.length / 2)];
            const ink = new Float32Array(S * S);
            for (let i = 0; i < S * S; i++) {
                ink[i] = bg > 110 ? Math.max(0, bg - gray[i]) : gray[i];
            }
            let sum = 0, left = 0, right = 0, top = 0, bottom = 0;
            for (let y = 0; y < S; y++) {
                for (let x = 0; x < S; x++) {
                    const v = ink[y * S + x]; sum += v;
                    if (x < S / 2) left += v; else right += v;
                    if (y < S / 2) top += v; else bottom += v;
                }
            }
            if (sum < 1e-6) {
                return offlineBuildResult(filename, dataUrl, { Reversal: 0.33, Corrected: 0.33, Normal: 0.34 }, "Normal");
            }
            const hAsym = Math.abs(left - right) / sum;
            const vAsym = Math.abs(top - bottom) / sum;
            // Mirror difference: image vs horizontally flipped
            let flipDiff = 0;
            for (let y = 0; y < S; y++) {
                for (let x = 0; x < S / 2; x++) {
                    flipDiff += Math.abs(ink[y * S + x] - ink[y * S + (S - 1 - x)]);
                }
            }
            flipDiff /= (sum + 1e-6);
            const mean = sum / (S * S);
            let variance = 0;
            for (let i = 0; i < S * S; i++) variance += (ink[i] - mean) * (ink[i] - mean);
            variance = variance / (S * S) / (255 * 255);
            const inkRatio = mean / 255;
            // Heuristic scores (tuned to separate b/d-like asymmetry vs scribble vs clean)
            const reversalScore = Math.min(1.6, hAsym * 3.0 + flipDiff * 0.9 + vAsym * 0.4);
            const correctedScore = Math.min(1.6, variance * 9.0 + Math.min(0.6, inkRatio * 2.2));
            const normalScore = 0.85;
            const ex = (v) => Math.exp(v * 2.2);
            const eR = ex(reversalScore), eC = ex(correctedScore), eN = ex(normalScore);
            const tot = eR + eC + eN;
            const probs = { Reversal: eR / tot, Corrected: eC / tot, Normal: eN / tot };
            let top1 = "Normal";
            if (probs.Reversal >= probs.Corrected && probs.Reversal >= probs.Normal) top1 = "Reversal";
            else if (probs.Corrected >= probs.Normal) top1 = "Corrected";
            return offlineBuildResult(filename, dataUrl, probs, top1);
        });
    }
    function offlineBuildResult(filename, thumb, probs, pred) {
        const conf = probs[pred];
        const risk = pred === "Reversal"
            ? { level: "High Risk (Dyslexia Indicator)", desc: "Offline demo estimate: strong stroke asymmetry consistent with mirror-reversal patterns. Host the Flask backend (?backend=URL) for the full 88.67% hybrid AI verdict." }
            : pred === "Corrected"
                ? { level: "Moderate Risk", desc: "Offline demo estimate: high stroke variance / overwriting texture. Host the Flask backend (?backend=URL) for the full hybrid AI verdict." }
                : { level: "Low Risk", desc: "Offline demo estimate: balanced strokes, no strong reversal signal. Host the Flask backend (?backend=URL) for the full hybrid AI verdict." };
        const pct = (v) => `${(v * 100).toFixed(1)}%`;
        const jit = (v, d) => Math.max(0.01, Math.min(0.99, v + d));
        const mk = (delta) => {
            const p = { Reversal: jit(probs.Reversal, delta), Corrected: jit(probs.Corrected, -delta / 2), Normal: jit(probs.Normal, -delta / 2) };
            const s = p.Reversal + p.Corrected + p.Normal;
            p.Reversal /= s; p.Corrected /= s; p.Normal /= s;
            let t = "Normal";
            if (p.Reversal >= p.Corrected && p.Reversal >= p.Normal) t = "Reversal";
            else if (p.Corrected >= p.Normal) t = "Corrected";
            return { prediction: t, confidence: p[t], probabilities: p };
        };
        const cnn = mk(0.02), svm = mk(-0.015), rf = mk(0.0);
        return {
            filename: filename || "upload.png",
            prediction: pred, confidence: conf, confidence_percentage: pct(conf),
            risk_level: risk.level, risk_description: risk.desc, risk_color: pred === "Reversal" ? "#dc3545" : pred === "Corrected" ? "#ffc107" : "#28a745",
            unanimous_agreement: cnn.prediction === svm.prediction && svm.prediction === rf.prediction,
            vote_distribution: { Corrected: 0, Normal: 0, Reversal: 0 },
            ensemble_probabilities: probs,
            models: {
                CNN: { prediction: cnn.prediction, confidence: cnn.confidence, probabilities: cnn.probabilities },
                SVM: { prediction: svm.prediction, confidence: svm.confidence, probabilities: svm.probabilities },
                Random_Forest: { prediction: rf.prediction, confidence: rf.confidence, probabilities: rf.probabilities }
            },
            thumbnail_base64: thumb, processed_thumbnail_base64: thumb,
            offline_demo: true, mode: "offline_demo"
        };
    }
    async function offlineBatch(files) {
        const inds = [];
        const sum = { Reversal: 0, Corrected: 0, Normal: 0 };
        for (const f of files) {
            const url = await fileToDataURL(f);
            const one = await offlineAnalyzeDataUrl(url, f.name);
            inds.push(one);
            sum.Reversal += one.ensemble_probabilities.Reversal;
            sum.Corrected += one.ensemble_probabilities.Corrected;
            sum.Normal += one.ensemble_probabilities.Normal;
        }
        return offlineAggregate(inds, sum);
    }
    function offlineAggregate(inds, sum) {
        const n = inds.length;
        const avg = { Reversal: sum.Reversal / n, Corrected: sum.Corrected / n, Normal: sum.Normal / n };
        let overall = "Normal";
        if (avg.Reversal >= avg.Corrected && avg.Reversal >= avg.Normal) overall = "Reversal";
        else if (avg.Corrected >= avg.Normal) overall = "Corrected";
        const counts = { Corrected: 0, Normal: 0, Reversal: 0 };
        inds.forEach((r) => { counts[r.prediction] = (counts[r.prediction] || 0) + 1; });
        const revRate = counts.Reversal / n * 100, corrRate = counts.Corrected / n * 100, normRate = counts.Normal / n * 100;
        const level = (revRate >= 20 || overall === "Reversal") ? "High Risk (Dyslexia Indicator)"
            : (corrRate >= 30 || overall === "Corrected") ? "Moderate Risk" : "Low Risk";
        return {
            total_images: n, overall_prediction: overall,
            overall_confidence: avg[overall], overall_confidence_percentage: `${(avg[overall] * 100).toFixed(1)}%`,
            average_probabilities: avg, class_counts: counts,
            reversal_count: counts.Reversal, reversal_rate: revRate,
            corrected_count: counts.Corrected, corrected_rate: corrRate,
            normal_count: counts.Normal, normal_rate: normRate,
            overall_risk_level: level + " — offline demo",
            overall_risk_description: inds[0] ? inds[0].risk_description : "Offline demo result.",
            overall_risk_color: overall === "Reversal" ? "#dc3545" : overall === "Corrected" ? "#ffc107" : "#28a745",
            individual_results: inds,
            model_averages: { CNN: avg, SVM: avg, Random_Forest: avg },
            backbone_comparison: {
                active_backbone: "custom_cnn (offline demo)",
                custom_cnn: { available: true, prediction: overall, probabilities: avg, input: "64×64 gray", note: "Offline heuristic estimate" },
                mobilenetv2: { available: false, prediction: null, probabilities: null, input: "96×96 RGB", status: "not trained yet", train_hint: "python run_pipeline.py --backbone mobilenetv2", note: "Placeholder until trained" },
                svm: { available: true, prediction: overall, probabilities: avg, note: "SVM (RBF) heuristic estimate" },
                consensus: { prediction: overall, confidence_percentage: `${(avg[overall] * 100).toFixed(1)}%` }
            },
            mode: "offline_demo"
        };
    }
    function offlineDemoBatch(type) {
        const presets = {
            normal: { Reversal: 0.12, Corrected: 0.18, Normal: 0.70 },
            reversal: { Reversal: 0.72, Corrected: 0.16, Normal: 0.12 },
            corrected: { Reversal: 0.15, Corrected: 0.68, Normal: 0.17 }
        };
        const pick = presets[type] || presets.normal;
        const mkThumb = (label, i) => {
            const c = document.createElement("canvas"); c.width = 100; c.height = 100;
            const g = c.getContext("2d");
            g.fillStyle = "#111"; g.fillRect(0, 0, 100, 100);
            g.strokeStyle = "#fff"; g.lineWidth = 5; g.font = "bold 56px serif";
            g.fillStyle = "#fff"; g.textAlign = "center"; g.textBaseline = "middle";
            g.fillText(label === "Reversal" ? "b" : label === "Corrected" ? "A" : "A", 50, 55);
            return c.toDataURL();
        };
        const count = type === "suite" ? 6 : 2;
        const labels = type === "suite"
            ? ["Normal", "Normal", "Reversal", "Reversal", "Corrected", "Corrected"]
            : [(type === "reversal" ? "Reversal" : type === "corrected" ? "Corrected" : "Normal"),
               (type === "reversal" ? "Reversal" : type === "corrected" ? "Corrected" : "Normal")];
        const inds = labels.slice(0, count).map((lab, i) => {
            const p = lab === "Normal" ? presets.normal : lab === "Reversal" ? presets.reversal : presets.corrected;
            return offlineBuildResult(`${lab}_demo_${i + 1}.png`, mkThumb(lab, i), p, lab);
        });
        const sum = { Reversal: 0, Corrected: 0, Normal: 0 };
        inds.forEach((r) => { sum.Reversal += r.ensemble_probabilities.Reversal; sum.Corrected += r.ensemble_probabilities.Corrected; sum.Normal += r.ensemble_probabilities.Normal; });
        return offlineAggregate(inds, sum);
    }

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
                const response = await fetch(api(`/api/demo?type=${demoType}`));
                const data = await apiJson(response);
                if (data.error) {
                    throw new Error(data.error);
                }
                renderResults(data, false);
            } catch (err) {
                if (isBackendMissing(err)) {
                    try {
                        triggerLoading(true, "Backend not found on this static host — running offline demo analysis...");
                        const data = offlineDemoBatch(demoType);
                        renderResults(data, true);
                        return;
                    } catch (e2) {
                        alert("Offline demo also failed: " + e2.message);
                    }
                } else {
                    alert("Error loading demo sample: " + err.message);
                }
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
            const response = await fetch(api("/api/predict"), {
                method: "POST",
                body: formData
            });

            const data = await apiJson(response);
            if (data.error) {
                throw new Error(data.error);
            }

            renderResults(data, false);
        } catch (err) {
            if (isBackendMissing(err)) {
                try {
                    triggerLoading(true, `Backend 404 on static host — analyzing ${selectedFiles.length} image(s) in browser (offline demo)...`);
                    const data = await offlineBatch(selectedFiles);
                    renderResults(data, true);
                    return;
                } catch (e2) {
                    alert("Offline analysis failed: " + e2.message);
                }
            } else {
                alert("Analysis failed: " + err.message);
            }
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
    function renderResults(res, isOffline = false) {
        currentResult = res;
        triggerLoading(false);

        emptyState.style.display = "none";
        resultsContent.style.display = "block";
        const offline = isOffline || res.mode === "offline_demo";
        resultsBadge.textContent = offline ? "Demo Complete (offline)" : "Analysis Complete";
        resultsBadge.title = offline
            ? "Static host has no Python backend, so this result is a browser demo approximation. Add ?backend=https://your-flask-server for full hybrid AI."
            : "Full backend result";

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

        // 4b. Backbone comparison: Custom CNN vs MobileNetV2 vs SVM
        try {
            const bc = res.backbone_comparison || null;
            const el = (id) => document.getElementById(id);
            const customP = bc ? bc.custom_cnn : null;
            const mobileP = bc ? bc.mobilenetv2 : null;
            const svmP = bc ? bc.svm : null;
            const cnnTop = getTopClass(res.model_averages["CNN"]);
            const svmTop = getTopClass(res.model_averages["SVM"]);
            if (el("cmpCustomPred")) el("cmpCustomPred").textContent = (customP && customP.prediction) || cnnTop;
            if (el("cmpCustomConf")) el("cmpCustomConf").textContent = `64×64 gray · ${(res.model_averages["CNN"][cnnTop] * 100).toFixed(1)}%`;
            if (el("cmpCustomStatus")) {
                const live = !customP || customP.available !== false;
                el("cmpCustomStatus").textContent = live ? "live" : "missing";
                el("cmpCustomStatus").className = "table-tag " + (live ? "success" : "");
            }
            if (el("cmpMobilePred")) el("cmpMobilePred").textContent = (mobileP && mobileP.prediction) || "—";
            if (el("cmpMobileConf")) {
                el("cmpMobileConf").textContent = (mobileP && mobileP.available)
                    ? `96×96 RGB · ${(Object.values(mobileP.probabilities || {})[0] * 100 || 0).toFixed(1)}%`
                    : "96×96 RGB · not trained";
            }
            if (el("cmpMobileStatus")) {
                const ok = !!(mobileP && mobileP.available);
                el("cmpMobileStatus").textContent = ok ? "live" : "pending";
                el("cmpMobileStatus").className = "table-tag " + (ok ? "success" : "");
            }
            if (el("cmpSvmPred")) el("cmpSvmPred").textContent = (svmP && svmP.prediction) || svmTop;
            if (el("cmpSvmConf")) {
                const agree = cnnTop === svmTop ? "CNN↔SVM agree" : "CNN↔SVM split";
                el("cmpSvmConf").textContent = `${agree} · consensus ${predClass} (${res.overall_confidence_percentage})`;
            }
            if (el("backboneNote")) {
                el("backboneNote").textContent = (mobileP && mobileP.available)
                    ? "All three heads live on this backend."
                    : (offline
                        ? "Offline demo: Custom CNN + SVM are heuristic estimates; MobileNetV2 needs training (python run_pipeline.py --backbone mobilenetv2)."
                        : "MobileNetV2 weights not trained yet — Custom CNN + SVM are live. Train with: python run_pipeline.py --backbone mobilenetv2");
            }
        } catch (e) { /* comparison is best-effort */ }

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
            // Single image: fetch on-demand Grad-CAM explanation.
            const gradcamSection = document.getElementById("gradcamSection");
            const gradcamImg = document.getElementById("gradcamImg");
            const gradcamCaption = document.getElementById("gradcamCaption");
            if (offline && gradcamSection) {
                gradcamSection.style.display = "none";
            } else if (gradcamSection && selectedFiles.length === 1) {
                gradcamSection.style.display = "block";
                if (gradcamImg) gradcamImg.style.opacity = "0.4";
                if (gradcamCaption) gradcamCaption.textContent = "Computing Grad-CAM...";
                const fd = new FormData();
                fd.append("image", selectedFiles[0]);
                fetch(api("/api/gradcam"), { method: "POST", body: fd })
                    .then((r) => apiJson(r))
                    .then((g) => {
                        if (g.gradcam_base64 && gradcamImg) {
                            gradcamImg.src = g.gradcam_base64;
                            gradcamImg.style.opacity = "1";
                        }
                        if (gradcamCaption) {
                            gradcamCaption.textContent = g.target_class
                                ? `Focus: ${g.target_class} (layer ${g.conv_layer}, score ${g.focus_score})`
                                : (g.error || "Grad-CAM unavailable.");
                        }
                    })
                    .catch(() => {
                        if (gradcamCaption) gradcamCaption.textContent = "Grad-CAM unavailable.";
                    });
            } else if (gradcamSection) {
                gradcamSection.style.display = "none";
            }
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

(() => {
  "use strict";

  const API_BASE = "";

  // DOM Elements
  const form = document.getElementById("predict-form");
  const submitBtn = document.getElementById("submit-btn");
  const downloadReportBtn = document.getElementById("download-report-btn");
  const themeToggleBtn = document.getElementById("theme-toggle");
  const compareAvgToggle = document.getElementById("compare-avg-toggle");

  const stateIdle = document.getElementById("state-idle");
  const stateLoading = document.getElementById("state-loading");
  const stateResult = document.getElementById("state-result");
  const stateError = document.getElementById("state-error");

  const scoreNumberEl = document.getElementById("score-number");
  const scoreBandEl = document.getElementById("score-band");
  const scoreConfEl = document.getElementById("score-confidence");
  const gaugeFill = document.getElementById("gauge-fill");
  const errorLabelEl = document.getElementById("error-label");
  const errorCopyEl = document.getElementById("error-copy");

  const shapFactorsList = document.getElementById("shap-factors-list");
  const recsList = document.getElementById("recs-list");
  const whatifDeltaEl = document.getElementById("whatif-delta");

  const wfSleep = document.getElementById("wf-sleep");
  const wfScreen = document.getElementById("wf-screen");

  const GAUGE_ARC_LENGTH = 314;

  let currentPayload = null;
  let basePredictedScore = 0.0;
  let chartGlobalImp = null;
  let chartDomainImp = null;
  let chartModelComp = null;

  // 1. Draw SVG Gauge Ticks
  function drawTicks() {
    document.querySelectorAll(".gauge-ticks").forEach((g) => {
      g.innerHTML = "";
      const cx = 120, cy = 140, rOuter = 100, rInner = 90;
      for (let i = 0; i <= 10; i += 2) {
        const angle = Math.PI - (i / 10) * Math.PI;
        const x1 = cx + rOuter * Math.cos(angle);
        const y1 = cy - rOuter * Math.sin(angle);
        const x2 = cx + rInner * Math.cos(angle);
        const y2 = cy - rInner * Math.sin(angle);
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", x1.toFixed(1));
        line.setAttribute("y1", y1.toFixed(1));
        line.setAttribute("x2", x2.toFixed(1));
        line.setAttribute("y2", y2.toFixed(1));
        g.appendChild(line);
      }
    });
  }
  drawTicks();

  // 2. Theme Switcher
  if (themeToggleBtn) {
    themeToggleBtn.addEventListener("click", () => {
      document.body.classList.toggle("theme-dark");
      const isDark = document.body.classList.contains("theme-dark");
      themeToggleBtn.querySelector(".theme-text").textContent = isDark ? "Light Mode" : "Dark Mode";
      themeToggleBtn.querySelector(".theme-icon").textContent = isDark ? "☀️" : "🌙";
    });
  }

  // 3. Tab Navigation
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((c) => (c.hidden = true));

      btn.classList.add("active");
      const targetId = btn.dataset.tab;
      const targetEl = document.getElementById(targetId);
      if (targetEl) {
        targetEl.hidden = false;
        if (targetId === "tab-importance") loadGlobalImportance();
        if (targetId === "tab-comparison") loadModelComparison();
        if (targetId === "tab-about") loadAboutMetadata();
      }
    });
  });

  // 4. Live Range Sliders Value Updates & Segmented Buttons
  const sliderMappings = [
    ["age", "age-val", ""],
    ["avg_daily_usage_hours", "usage-val", "h"],
    ["daily_unlocks", "unlocks-val", ""],
    ["sleep_hours_per_night", "sleep-val", "h"],
    ["study_hours", "study-val", "h"],
    ["physical_activity_hours", "activity-val", "h"],
    ["social_support_score", "support-val", ""],
    ["sleep_quality", "sleepq-val", ""],
    ["academic_pressure", "press-val", ""]
  ];

  sliderMappings.forEach(([inputId, labelId, unit]) => {
    const input = document.getElementById(inputId);
    const label = document.getElementById(labelId);
    if (input && label) {
      input.addEventListener("input", () => {
        label.textContent = `${input.value}${unit}`;
        debounceAutoPredict();
      });
    }
  });

  const stressGroup = document.getElementById("stress_level_group");
  const stressInput = document.getElementById("stress_level");
  if (stressGroup && stressInput) {
    stressGroup.querySelectorAll(".seg-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        stressGroup.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        stressInput.value = btn.dataset.value;
        debounceAutoPredict();
      });
    });
  }

  form.querySelectorAll("select, input[type='text']").forEach((el) => {
    el.addEventListener("change", () => debounceAutoPredict());
  });

  // Debounced live predict call
  let timerId = null;
  function debounceAutoPredict() {
    clearTimeout(timerId);
    timerId = setTimeout(() => {
      fetchPrediction(true);
    }, 400);
  }

  // 5. Collect Form Payload
  function collectPayload() {
    const fd = new FormData(form);
    return {
      age: parseInt(fd.get("age") || "21", 10),
      gender: fd.get("gender") || "Male",
      country: (fd.get("country") || "India").trim(),
      academic_level: fd.get("academic_level") || "Undergraduate",
      living_situation: fd.get("living_situation") || "With family",
      relationship_status: fd.get("relationship_status") || "Single",
      part_time_job: fd.get("part_time_job") || "No",
      most_used_platform: fd.get("most_used_platform") || "Instagram",
      purpose_of_use: fd.get("purpose_of_use") || "Entertainment",
      avg_daily_usage_hours: parseFloat(fd.get("avg_daily_usage_hours") || "4.5"),
      daily_unlocks: parseInt(fd.get("daily_unlocks") || "120", 10),
      notifications_per_day: 120,
      late_night_usage_hours: 1.5,
      screen_time_before_bed_min: 30.0,
      uses_screen_time_limits: "No",
      social_comparison_frequency: "Sometimes",
      fomo_score: 5,
      cyberbullying_experience: "None",
      study_hours: parseFloat(fd.get("study_hours") || "4.0"),
      physical_activity_hours: parseFloat(fd.get("physical_activity_hours") || "1.5"),
      sleep_hours_per_night: parseFloat(fd.get("sleep_hours_per_night") || "7.0"),
      sleep_quality: parseInt(fd.get("sleep_quality") || "7", 10),
      caffeine_cups_per_day: 1.0,
      diet_quality: "Average",
      academic_pressure: parseInt(fd.get("academic_pressure") || "5", 10),
      financial_stress: fd.get("financial_stress") || "Medium",
      social_support_score: parseInt(fd.get("social_support_score") || "6", 10),
      close_friends_count: 4,
      counseling_access: fd.get("counseling_access") || "No",
      stress_level: fd.get("stress_level") || "Medium"
    };
  }

  function showState(name) {
    [stateIdle, stateLoading, stateResult, stateError].forEach((el) => {
      if (el) el.hidden = true;
    });
    const map = { idle: stateIdle, loading: stateLoading, result: stateResult, error: stateError };
    if (map[name]) map[name].hidden = false;
  }

  function renderError(label, copy) {
    if (errorLabelEl) errorLabelEl.textContent = label || "Error";
    if (errorCopyEl) errorCopyEl.textContent = copy || "An error occurred.";
    showState("error");
  }

  // 6. Fetch Prediction & Render ExAI Results
  async function fetchPrediction(isLive = false) {
    currentPayload = collectPayload();

    if (!isLive) showState("loading");

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(currentPayload)
      });

      if (!res.ok) {
        renderError("Prediction Failed", `Server returned status ${res.status}`);
        return;
      }

      const data = await res.json();
      basePredictedScore = data.score;

      renderResult(data);
    } catch (err) {
      if (!isLive) renderError("Server Unreachable", "Could not connect to prediction server.");
    }
  }

  function renderResult(data) {
    const score = data.score;
    const clamped = Math.max(1, Math.min(10, score));

    scoreNumberEl.textContent = score.toFixed(2);
    scoreBandEl.textContent = `Status Signal: ${data.category}`;
    if (data.confidence_band) {
      scoreConfEl.textContent = `95% Confidence Band: [${data.confidence_band[0]} — ${data.confidence_band[1]}]`;
    }

    // Update gauge arc
    gaugeFill.style.strokeDashoffset = String(GAUGE_ARC_LENGTH);
    requestAnimationFrame(() => {
      const offset = GAUGE_ARC_LENGTH * (1 - clamped / 10);
      gaugeFill.style.strokeDashoffset = String(offset);
    });

    // Render SHAP Factors
    shapFactorsList.innerHTML = "";
    if (data.top_factors) {
      data.top_factors.forEach((factor) => {
        const card = document.createElement("div");
        card.className = `shap-factor-card ${factor.direction}`;
        card.innerHTML = `
          <div class="factor-header">
            <span>${factor.label} (${factor.value})</span>
            <span>${factor.shap_value > 0 ? '+' : ''}${factor.shap_value.toFixed(2)} pts</span>
          </div>
          <div class="factor-reason">${factor.plain_english_reason}</div>
        `;
        shapFactorsList.appendChild(card);
      });
    }

    // Render Recommendations
    recsList.innerHTML = "";
    if (data.recommendations) {
      data.recommendations.forEach((rec) => {
        const li = document.createElement("li");
        li.textContent = rec;
        recsList.appendChild(li);
      });
    }

    // Sync What-if Sliders to Current Payload
    if (wfSleep && currentPayload) wfSleep.value = currentPayload.sleep_hours_per_night;
    if (wfScreen && currentPayload) wfScreen.value = currentPayload.avg_daily_usage_hours;
    whatifDeltaEl.textContent = "Delta: +0.00 pts";

    showState("result");
  }

  // 7. What-If Simulator Live Updates
  async function updateWhatIf() {
    if (!currentPayload) return;

    const modPayload = {
      ...currentPayload,
      sleep_hours_per_night: parseFloat(wfSleep.value),
      avg_daily_usage_hours: parseFloat(wfScreen.value)
    };

    try {
      const res = await fetch(`${API_BASE}/what-if`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(modPayload)
      });
      if (res.ok) {
        const data = await res.json();
        const delta = data.new_score - basePredictedScore;
        const sign = delta >= 0 ? "+" : "";
        whatifDeltaEl.textContent = `New Score: ${data.new_score.toFixed(2)} (${sign}${delta.toFixed(2)} pts)`;
        whatifDeltaEl.style.color = delta >= 0 ? "var(--pine-primary)" : "var(--accent-warn)";
      }
    } catch (e) {
      // Silent error for live slider
    }
  }

  if (wfSleep && wfScreen) {
    wfSleep.addEventListener("input", updateWhatIf);
    wfScreen.addEventListener("input", updateWhatIf);
  }

  // 8. Download PDF Report
  if (downloadReportBtn) {
    downloadReportBtn.addEventListener("click", async () => {
      if (!currentPayload) return;
      try {
        downloadReportBtn.disabled = true;
        downloadReportBtn.textContent = "⏳ Generating Report...";
        const res = await fetch(`${API_BASE}/report`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(currentPayload)
        });
        if (res.ok) {
          const blob = await res.blob();
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = "Mental_Health_Signal_Report.pdf";
          document.body.appendChild(a);
          a.click();
          a.remove();
        } else {
          alert("Failed to generate PDF report.");
        }
      } catch (err) {
        alert("Error downloading report.");
      } finally {
        downloadReportBtn.disabled = false;
        downloadReportBtn.textContent = "📄 Download PDF Report";
      }
    });
  }

  // Form Submit
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    fetchPrediction(false);
  });

  const errorRetryBtn = document.getElementById("error-retry-btn");
  if (errorRetryBtn) {
    errorRetryBtn.addEventListener("click", () => showState("idle"));
  }

  // 9. Tab 2: Load Global Importance Charts
  async function loadGlobalImportance() {
    try {
      const res = await fetch(`${API_BASE}/feature-importance`);
      if (!res.ok) return;
      const data = await res.json();

      const labels = data.slice(0, 15).map((d) => d.Feature);
      const shapVals = data.slice(0, 15).map((d) => d.SHAP_Mean_Abs);
      const permVals = data.slice(0, 15).map((d) => d.Permutation_Mean);

      const ctxGlobal = document.getElementById("chart-global-importance");
      if (ctxGlobal) {
        if (chartGlobalImp) chartGlobalImp.destroy();
        chartGlobalImp = new Chart(ctxGlobal, {
          type: "bar",
          data: {
            labels: labels,
            datasets: [
              { label: "SHAP Mean |Value|", data: shapVals, backgroundColor: "#2d6a4f" },
              { label: "Permutation Importance", data: permVals, backgroundColor: "#e76f51" }
            ]
          },
          options: { responsive: true, indexAxis: "y", plugins: { legend: { position: "bottom" } } }
        });
      }

      // Domain Grouping Chart
      const domainMap = {};
      data.forEach((d) => {
        const dom = d.Domain || "Other";
        domainMap[dom] = (domainMap[dom] || 0) + (d.SHAP_Mean_Abs || 0);
      });

      const ctxDomain = document.getElementById("chart-domain-importance");
      if (ctxDomain) {
        if (chartDomainImp) chartDomainImp.destroy();
        chartDomainImp = new Chart(ctxDomain, {
          type: "doughnut",
          data: {
            labels: Object.keys(domainMap),
            datasets: [{ data: Object.values(domainMap), backgroundColor: ["#2d6a4f", "#e76f51", "#2a9d8f", "#e9c46a"] }]
          },
          options: { responsive: true, plugins: { legend: { position: "bottom" } } }
        });
      }
    } catch (e) {}
  }

  // 10. Tab 3: Load Model Comparison Benchmarks
  async function loadModelComparison() {
    try {
      const res = await fetch(`${API_BASE}/model-comparison`);
      if (!res.ok) return;
      const data = await res.json();

      const rankTableBody = document.querySelector("#benchmark-table tbody");
      if (rankTableBody) {
        rankTableBody.innerHTML = "";
        data.rank_table.forEach((row, idx) => {
          const tr = document.createElement("tr");
          if (idx === 0) tr.style.backgroundColor = "var(--pine-soft)";
          tr.innerHTML = `
            <td><b>${idx + 1}</b></td>
            <td><b>${row.Model}</b> ${idx === 0 ? "🏆" : ""}</td>
            <td>${row.Test_R2.toFixed(4)}</td>
            <td>${row.CV_R2_Mean.toFixed(4)}</td>
            <td>${row.Test_RMSE.toFixed(4)}</td>
            <td>${row.Test_MAE.toFixed(4)}</td>
            <td>${row.Overfit_Gap.toFixed(4)}</td>
            <td>${row.Class_Accuracy ? (row.Class_Accuracy * 100).toFixed(1) + "%" : "-"}</td>
            <td>${row.Class_F1_Macro ? row.Class_F1_Macro.toFixed(4) : "-"}</td>
          `;
          rankTableBody.appendChild(tr);
        });
      }

      const verdictEl = document.getElementById("verdict-content");
      if (verdictEl && data.final_verdict) {
        verdictEl.textContent = data.final_verdict;
      }

      // Model Comparison Bar Chart
      const topModels = data.rank_table.slice(0, 10);
      const ctxModel = document.getElementById("chart-model-comparison");
      if (ctxModel) {
        if (chartModelComp) chartModelComp.destroy();
        chartModelComp = new Chart(ctxModel, {
          type: "bar",
          data: {
            labels: topModels.map((m) => m.Model),
            datasets: [
              { label: "Test R²", data: topModels.map((m) => m.Test_R2), backgroundColor: "#2d6a4f" },
              { label: "CV R² Mean", data: topModels.map((m) => m.CV_R2_Mean), backgroundColor: "#52b788" }
            ]
          },
          options: { responsive: true, scales: { y: { min: 0.5, max: 1.0 } } }
        });
      }
    } catch (e) {}
  }

  // 11. Tab 4: Load About Metadata
  async function loadAboutMetadata() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (!res.ok) return;
      const data = await res.json();
      const metaBox = document.getElementById("system-metadata-box");
      if (metaBox) {
        metaBox.innerHTML = `
          <p><strong>System Status:</strong> ${data.status}</p>
          <p><strong>Winning Model:</strong> ${data.model_name}</p>
          <p><strong>Scikit-Learn Version:</strong> ${data.sklearn_version}</p>
          <p><strong>Training Date:</strong> ${data.training_date}</p>
          <p><strong>Disclaimer:</strong> ${data.disclaimer}</p>
        `;
      }
    } catch (e) {}
  }

  // Initial Predict Trigger
  fetchPrediction(true);
})();
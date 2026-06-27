// Cost of Living Explainer — frontend logic (vanilla JS).
const API = "/api";
let STATE = { cities: [], regions: [], labels: {} };

const $ = (sel) => document.querySelector(sel);
const fmtMoney = (n) => "$" + Math.round(n).toLocaleString();

async function jget(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}
async function jpost(url, body) {
  const r = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

// ---- tabs -----------------------------------------------------------------
document.querySelectorAll(".tab").forEach((t) => {
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    $("#" + t.dataset.tab).classList.add("active");
    if (t.dataset.tab === "insights") loadInsights();
  });
});

// ---- driver bar chart (SHAP) ---------------------------------------------
function renderDrivers(container, drivers, maxAbs) {
  const max = maxAbs || Math.max(...drivers.map((d) => Math.abs(d.contribution)), 1);
  container.innerHTML = drivers.map((d) => {
    const pct = Math.min(100, (Math.abs(d.contribution) / max) * 50);
    const pos = d.contribution >= 0;
    const sign = pos ? "+" : "";
    return `<div class="driver-row">
      <div class="driver-name">${d.label}</div>
      <div class="driver-track"><div class="driver-mid"></div>
        <div class="driver-fill ${pos ? "pos" : "neg"}" style="width:${pct}%"></div>
      </div>
      <div class="driver-val ${pos ? "pos" : "neg"}">${sign}${d.contribution.toFixed(1)}</div>
    </div>`;
  }).join("");
}

// ---- EXPLORE --------------------------------------------------------------
function burdenLabel(b) {
  if (b >= 130) return "very high";
  if (b >= 105) return "high";
  if (b >= 85) return "moderate";
  return "comfortable";
}

async function loadCity(name) {
  const d = await jget(`${API}/city/${encodeURIComponent(name)}`);
  $("#cityIndex").textContent = d.explanation.predicted_index.toFixed(1);
  $("#cityIndexSub").textContent = `actual ${d.actual_index.toFixed(1)} · ${d.region}`;
  $("#cityIncome").textContent = fmtMoney(d.median_income_usd);
  $("#cityBurden").textContent = d.affordability_burden.toFixed(0);
  $("#cityBurdenSub").textContent = `${burdenLabel(d.affordability_burden)} — cost vs wages`;
  renderDrivers($("#driverChart"), d.explanation.drivers);

  $("#similarList").innerHTML = d.similar_cities.map((s) => `
    <div class="similar-item" data-city="${s.city}">
      <div><div class="name">${s.city}</div><div class="region">${s.region}</div></div>
      <div class="idx">${s.cost_of_living_index.toFixed(0)}</div>
    </div>`).join("");
  document.querySelectorAll("#similarList .similar-item").forEach((el) =>
    el.addEventListener("click", () => {
      $("#citySelect").value = el.dataset.city;
      loadCity(el.dataset.city);
    }));
}

// ---- COMPARE --------------------------------------------------------------
async function loadCompare() {
  const a = $("#cmpA").value, b = $("#cmpB").value;
  if (!a || !b) return;
  const d = await jget(`${API}/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`);
  $("#cmpHeads").innerHTML = [d.a, d.b].map((c, i) => `
    <div class="card metric">
      <div class="metric-label">City ${i === 0 ? "A" : "B"} · ${c.city}</div>
      <div class="metric-value">${c.index.toFixed(1)}</div>
      <div class="metric-sub">income ${fmtMoney(c.income)}/mo</div>
    </div>`).join("");

  const max = Math.max(...d.category_gaps.map((g) => Math.abs(g.gap)), 1);
  $("#cmpChart").innerHTML = d.category_gaps.map((g) => {
    const pct = Math.min(100, (Math.abs(g.gap) / max) * 50);
    const pos = g.gap >= 0;
    return `<div class="dv-row">
      <div class="driver-name">${g.label}</div>
      <div class="dv-track"><div class="dv-mid"></div>
        <div class="dv-fill ${pos ? "pos" : "neg"}" style="width:${pct}%"></div></div>
      <div class="driver-val">${pos ? "+" : ""}${g.gap.toFixed(0)}</div>
    </div>`;
  }).join("");
}

// ---- INSIGHTS -------------------------------------------------------------
let insightsLoaded = false;
async function loadInsights() {
  if (insightsLoaded) return;
  const d = await jget(`${API}/insights`);
  const max = Math.max(...d.global_importance.map((x) => x.importance), 1);
  $("#globalChart").innerHTML = d.global_importance.map((x) => {
    const pct = Math.min(100, (x.importance / max) * 100);
    return `<div class="driver-row">
      <div class="driver-name">${x.label}</div>
      <div class="driver-track">
        <div class="driver-fill pos" style="left:0;width:${pct}%"></div></div>
      <div class="driver-val pos">${x.importance.toFixed(1)}</div>
    </div>`;
  }).join("");

  $("#regionTable").innerHTML = `<table><thead><tr>
    <th>Region</th><th class="num">Cost idx</th><th class="num">Income</th><th class="num">Burden</th>
    </tr></thead><tbody>${d.regions.map((r) => `<tr>
      <td>${r.region}</td><td class="num">${r.avg_index.toFixed(0)}</td>
      <td class="num">${fmtMoney(r.avg_income)}</td><td class="num">${r.avg_burden.toFixed(0)}</td>
    </tr>`).join("")}</tbody></table>`;

  $("#modelMetrics").innerHTML = `
    <div class="m"><b>${d.metrics.r2}</b><span>R² (hold-out)</span></div>
    <div class="m"><b>${d.metrics.mae}</b><span>MAE (index pts)</span></div>
    <div class="m"><b>${d.metrics.n_train + d.metrics.n_test}</b><span>cities</span></div>`;
  insightsLoaded = true;
}

// ---- PREDICT --------------------------------------------------------------
const SLIDERS = [
  ["housing_index", "Housing & rent", 0, 250, 100],
  ["groceries_index", "Groceries", 0, 200, 100],
  ["transport_index", "Transport", 0, 200, 100],
  ["utilities_index", "Utilities", 0, 200, 100],
  ["restaurant_index", "Restaurants", 0, 200, 100],
  ["healthcare_index", "Healthcare", 0, 200, 100],
  ["childcare_index", "Childcare", 0, 200, 100],
  ["median_income_usd", "Income $/mo", 300, 12000, 3000],
  ["population_density", "Density /km²", 300, 25000, 5000],
];

function buildSliders() {
  $("#predictForm").innerHTML = SLIDERS.map(([k, lbl, lo, hi, def]) => `
    <div class="srow">
      <label class="sname">${lbl}</label>
      <input type="range" id="s_${k}" min="${lo}" max="${hi}" value="${def}"
        step="${k === "median_income_usd" ? 100 : k === "population_density" ? 100 : 1}" />
      <span class="sval" id="v_${k}">${def}</span>
    </div>`).join("");
  SLIDERS.forEach(([k]) => {
    const inp = $(`#s_${k}`);
    inp.addEventListener("input", () => { $(`#v_${k}`).textContent = inp.value; });
  });
}

async function runPredict() {
  const body = { region: $("#predRegion").value, tourism_intensity: 0.5 };
  SLIDERS.forEach(([k]) => { body[k] = parseFloat($(`#s_${k}`).value); });
  const d = await jpost(`${API}/predict`, body);
  $("#predIndex").textContent = d.predicted_index.toFixed(1);
  renderDrivers($("#predDrivers"), d.drivers);
}

// ---- init -----------------------------------------------------------------
function fillSelect(sel, values, selected) {
  sel.innerHTML = values.map((v) => `<option ${v === selected ? "selected" : ""}>${v}</option>`).join("");
}

async function init() {
  try {
    const h = await jget(`${API}/health`);
    $("#modelBadge").textContent = h.model_loaded
      ? `model ready · R² ${h.metrics.r2}` : "model not trained — run scripts.train";
  } catch { $("#modelBadge").textContent = "API offline"; }

  const data = await jget(`${API}/cities`);
  STATE.cities = data.cities.map((c) => c.city);
  STATE.regions = data.regions;
  STATE.labels = data.feature_labels;

  fillSelect($("#citySelect"), STATE.cities, STATE.cities[0]);
  fillSelect($("#cmpA"), STATE.cities, STATE.cities[0]);
  fillSelect($("#cmpB"), STATE.cities, STATE.cities[1]);
  fillSelect($("#predRegion"), STATE.regions, "North America");

  $("#citySelect").addEventListener("change", (e) => loadCity(e.target.value));
  $("#cmpA").addEventListener("change", loadCompare);
  $("#cmpB").addEventListener("change", loadCompare);
  $("#predictBtn").addEventListener("click", runPredict);

  buildSliders();
  await loadCity(STATE.cities[0]);
  await loadCompare();
}

init().catch((e) => {
  document.querySelector("main").insertAdjacentHTML("afterbegin",
    `<div class="card" style="border-color:#ff6b6b">Failed to load: ${e.message}</div>`);
});

// Cost of Living Explainer — frontend logic (vanilla JS). New Delhi = 100 baseline.
const API = "/api";
let STATE = { cities: [], regions: [] };
let MAP = null, MARKERS = {}, SELECTED_ID = null;

const $ = (s) => document.querySelector(s);
const fmt = (n) => Number(n).toLocaleString(undefined, { maximumFractionDigits: 0 });
const mult = (v) => (v / 100 >= 1 ? (v / 100).toFixed(1) + "×" : (v / 100).toFixed(2) + "×");

async function jget(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}
async function jpost(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
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

// ---- driver bars ----------------------------------------------------------
function renderDrivers(container, drivers, maxAbs) {
  const max = maxAbs || Math.max(...drivers.map((d) => Math.abs(d.contribution)), 1);
  container.innerHTML = drivers.map((d) => {
    const pct = Math.min(100, (Math.abs(d.contribution) / max) * 50);
    const pos = d.contribution >= 0;
    return `<div class="driver-row">
      <div class="driver-name">${d.label}</div>
      <div class="driver-track"><div class="driver-mid"></div>
        <div class="driver-fill ${pos ? "pos" : "neg"}" style="width:${pct}%"></div></div>
      <div class="driver-val ${pos ? "pos" : "neg"}">${pos ? "+" : ""}${d.contribution.toFixed(1)}</div>
    </div>`;
  }).join("");
}

// ---- EXPLORE --------------------------------------------------------------
async function loadCity(id) {
  const d = await jget(`${API}/city?id=${encodeURIComponent(id)}`);
  $("#ovTitle").textContent = `${d.city}, ${d.country}`;
  $("#cityOverview").textContent = d.overview;

  $("#cityIndex").textContent = d.total_cost_index.toFixed(0);
  $("#cityIndexSub").innerHTML = `<b>${mult(d.total_cost_index)}</b> New Delhi · ${d.region}`;
  $("#cityRent").textContent = d.rent_index.toFixed(0);
  $("#cityRentSub").innerHTML = `<b>${mult(d.rent_index)}</b> New Delhi`;
  $("#cityPP").textContent = d.purchasing_power_index.toFixed(0);
  const ppCool = d.purchasing_power_index >= 100;
  $("#cityPPSub").innerHTML = `<b class="${ppCool ? "cool" : ""}">${mult(d.purchasing_power_index)}</b> New Delhi — wages stretch ${ppCool ? "further" : "less"}`;

  renderDrivers($("#driverChart"), d.explanation.drivers);

  $("#similarList").innerHTML = d.similar_cities.map((s) => `
    <div class="similar-item" data-id="${s.city}|${s.country}">
      <div><div class="name">${s.city}</div><div class="region">${s.country} · ${s.region}</div></div>
      <div class="idx">${s.total_cost_index.toFixed(0)}</div>
    </div>`).join("");
  document.querySelectorAll("#similarList .similar-item").forEach((el) =>
    el.addEventListener("click", () => selectCity(el.dataset.id, true)));
}

// ---- COMPARE --------------------------------------------------------------
async function loadCompare() {
  const a = $("#cmpA").value, b = $("#cmpB").value;
  if (!a || !b) return;
  const d = await jget(`${API}/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`);
  $("#cmpHeads").innerHTML = [d.a, d.b].map((c, i) => `
    <div class="card metric">
      <div class="metric-label">City ${i === 0 ? "A" : "B"} · ${c.city}, ${c.country}</div>
      <div class="metric-value">${c.index.toFixed(0)}</div>
      <div class="metric-sub">purchasing power <b class="${c.purchasing_power >= 100 ? "cool" : ""}">${c.purchasing_power.toFixed(0)}</b></div>
    </div>`).join("");

  const max = Math.max(...d.category_gaps.map((g) => Math.abs(g.gap)), 1);
  $("#cmpChart").innerHTML = d.category_gaps.map((g) => {
    const pct = Math.min(100, (Math.abs(g.gap) / max) * 50);
    const pos = g.gap >= 0;
    return `<div class="dv-row">
      <div class="driver-name">${g.label}</div>
      <div class="dv-track"><div class="dv-mid"></div>
        <div class="dv-fill ${pos ? "pos" : "neg"}" style="width:${pct}%"></div></div>
      <div class="driver-val ${pos ? "pos" : "neg"}">${pos ? "+" : ""}${g.gap.toFixed(0)}</div>
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
      <div class="driver-track"><div class="driver-fill pos" style="left:0;width:${pct}%"></div></div>
      <div class="driver-val pos">${x.importance.toFixed(1)}</div>
    </div>`;
  }).join("");

  $("#regionTable").innerHTML = `<table><thead><tr>
    <th>Region</th><th class="num">Cost</th><th class="num">Rent</th><th class="num">Purch. power</th><th class="num">n</th>
    </tr></thead><tbody>${d.regions.map((r) => `<tr>
      <td>${r.region}</td><td class="num">${r.avg_total.toFixed(0)}</td>
      <td class="num">${r.avg_rent.toFixed(0)}</td><td class="num">${r.avg_purchasing_power.toFixed(0)}</td>
      <td class="num">${r.n}</td></tr>`).join("")}</tbody></table>`;

  const m = d.metrics;
  $("#modelMetrics").innerHTML = `
    <div class="m"><b>${m.r2}</b><span>R² (hold-out)</span></div>
    <div class="m"><b>${m.mae}</b><span>MAE (index pts)</span></div>
    <div class="m"><b>${m.n_cities}</b><span>cities</span></div>
    <div class="m"><b>${m.n_countries}</b><span>countries</span></div>`;
  insightsLoaded = true;
}

// ---- PREDICT --------------------------------------------------------------
const SLIDERS = [
  ["rent_index", "Rent & housing", 0, 1500, 100, 5],
  ["groceries_index", "Groceries", 0, 600, 100, 2],
  ["restaurant_index", "Restaurants", 0, 600, 100, 2],
  ["purchasing_power_index", "Local purchasing power", 0, 400, 100, 2],
];
function buildSliders() {
  $("#predictForm").innerHTML = SLIDERS.map(([k, lbl, lo, hi, def, step]) => `
    <div class="srow">
      <label class="sname">${lbl}</label>
      <input type="range" id="s_${k}" min="${lo}" max="${hi}" value="${def}" step="${step}" />
      <span class="sval" id="v_${k}">${def}</span>
    </div>`).join("");
  SLIDERS.forEach(([k]) => {
    const inp = $(`#s_${k}`);
    inp.addEventListener("input", () => { $(`#v_${k}`).textContent = inp.value; });
  });
}
async function runPredict() {
  const body = { region: $("#predRegion").value };
  SLIDERS.forEach(([k]) => { body[k] = parseFloat($(`#s_${k}`).value); });
  const d = await jpost(`${API}/predict`, body);
  $("#predIndex").textContent = d.predicted_index.toFixed(0);
  renderDrivers($("#predDrivers"), d.drivers);
}

// ---- MAP ------------------------------------------------------------------
function costColor(idx) {
  const stops = [[0, [47, 143, 134]], [0.5, [216, 178, 90]], [1, [176, 83, 46]]];
  const t = Math.max(0, Math.min(1, (idx - 100) / (450 - 100)));
  let a = stops[0], b = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++)
    if (t >= stops[i][0] && t <= stops[i + 1][0]) { a = stops[i]; b = stops[i + 1]; break; }
  const lt = (t - a[0]) / ((b[0] - a[0]) || 1);
  const c = a[1].map((v, i) => Math.round(v + (b[1][i] - v) * lt));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function initMap(cities) {
  MAP = L.map("map", { worldCopyJump: true, minZoom: 2, maxZoom: 12, scrollWheelZoom: true })
    .setView([28, 12], 2);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
    attribution: '&copy; OpenStreetMap, &copy; CARTO', subdomains: "abcd", maxZoom: 19,
  }).addTo(MAP);

  cities.forEach((c) => {
    if (c.lat == null || c.lng == null) return;
    const m = L.circleMarker([c.lat, c.lng], {
      radius: 5, color: "#2a2e2a", weight: 1,
      fillColor: costColor(c.total_cost_index), fillOpacity: 0.85,
    });
    m.bindTooltip(
      `<span class="city-tip"><b>${c.city}</b>, ${c.country} · cost <span>${c.total_cost_index.toFixed(0)}</span></span>`,
      { direction: "top", offset: [0, -4] });
    m.on("click", () => selectCity(c.id, false));
    m.addTo(MAP);
    MARKERS[c.id] = m;
  });
}

function highlightMarker(id, pan) {
  if (SELECTED_ID && MARKERS[SELECTED_ID])
    MARKERS[SELECTED_ID].setStyle({ radius: 5, color: "#2a2e2a", weight: 1 });
  const m = MARKERS[id];
  if (m) {
    m.setStyle({ radius: 8, color: "#1d6f6a", weight: 2.5 });
    m.bringToFront();
    if (pan && MAP) MAP.panTo(m.getLatLng(), { animate: true });
  }
  SELECTED_ID = id;
}

// Single entry point for choosing a city (map, dropdown, similar list all use it).
async function selectCity(id, pan = false) {
  $("#citySelect").value = id;
  highlightMarker(id, pan);
  await loadCity(id);
}

// ---- init -----------------------------------------------------------------
function cityOptions(cities) {
  return cities.map((c) => `<option value="${c.id}">${c.city}, ${c.country}</option>`).join("");
}
function fillRegions(sel, regions, selected) {
  sel.innerHTML = regions.map((r) => `<option ${r === selected ? "selected" : ""}>${r}</option>`).join("");
}

async function init() {
  try {
    const h = await jget(`${API}/health`);
    $("#modelBadge").innerHTML = h.model_loaded
      ? `R² <b>${h.metrics.r2}</b> · ${h.metrics.n_cities} cities`
      : "model not trained";
  } catch { $("#modelBadge").textContent = "API offline"; }

  const data = await jget(`${API}/cities`);
  STATE.cities = data.cities;
  STATE.regions = data.regions;

  const opts = cityOptions(data.cities);
  $("#citySelect").innerHTML = opts;
  $("#cmpA").innerHTML = opts;
  $("#cmpB").innerHTML = opts;
  // sensible defaults
  const find = (name) => (data.cities.find((c) => c.city === name) || data.cities[0]).id;
  $("#citySelect").value = find("Delhi");
  $("#cmpA").value = find("Mumbai");
  $("#cmpB").value = find("Zurich");
  fillRegions($("#predRegion"), data.regions, "South Asia");

  $("#citySelect").addEventListener("change", (e) => selectCity(e.target.value, true));
  $("#cmpA").addEventListener("change", loadCompare);
  $("#cmpB").addEventListener("change", loadCompare);
  $("#predictBtn").addEventListener("click", runPredict);

  buildSliders();
  initMap(data.cities);
  await selectCity($("#citySelect").value, true);
  await loadCompare();
}

init().catch((e) => {
  document.querySelector("main").insertAdjacentHTML("afterbegin",
    `<div class="card" style="border-color:#b0532e">Failed to load: ${e.message}</div>`);
});

// Data: python -m evaluation.export_site_data (walk-forward, no look-ahead).
const $ = (id) => document.getElementById(id);
const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
const NAMES = { "^GSPC": "S&P 500", AAPL: "Apple", MSFT: "Microsoft", IBM: "IBM" };
const W = 1000, H = 420;
let DATA = {}, cur = null;

function draw(t) {
  cur = DATA[t];
  document.querySelectorAll("#tickers button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.t === t)));
  const all = cur.strategy.concat(cur.hold);
  const lo = Math.log10(Math.min(...all) * 0.9), hi = Math.log10(Math.max(...all) * 1.1);
  const n = cur.dates.length;
  const X = (i) => (i / (n - 1)) * W, Y = (v) => H - ((Math.log10(v) - lo) / (hi - lo)) * H;
  const path = (arr) => arr.map((v, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join("");
  let svg = "";
  const ticks = [];
  for (let e = Math.floor(lo); e <= Math.ceil(hi); e++) for (const m of [1, 3]) {
    const v = m * 10 ** e;
    if (Math.log10(v) >= lo && Math.log10(v) <= hi) ticks.push(v);
  }
  ticks.forEach((v) => (svg += `<line class="grid" x1="0" x2="${W}" y1="${Y(v)}" y2="${Y(v)}"/>`));
  svg += `<path class="l hold" d="${path(cur.hold)}"/><path class="l strat" d="${path(cur.strategy)}"/>`;
  svg += `<line class="cross" id="cross" x1="0" x2="0" y1="0" y2="${H}" visibility="hidden"/>`;
  $("chart").innerHTML = svg;

  const wrap = document.querySelector(".chart-wrap");
  wrap.querySelectorAll(".ylab,.xlab").forEach((e) => e.remove());
  ticks.forEach((v) => wrap.insertAdjacentHTML("beforeend", `<span class="ylab" style="top:${(Y(v) / H) * 100}%">$${v >= 1 ? v : v.toFixed(1)}</span>`));
  [2000, 2005, 2010, 2015, 2020, 2025].forEach((y) => {
    const i = cur.dates.findIndex((d) => d.startsWith(String(y)));
    if (i >= 0) wrap.insertAdjacentHTML("beforeend", `<span class="xlab" style="left:${(i / (n - 1)) * 100}%">${y}</span>`);
  });

  $("acc").textContent = pct(cur.accuracy);
  $("acc-l").innerHTML = `next-day accuracy, vs <b>${pct(cur.always_up)}</b> for always guessing up`;
  $("cagr").textContent = pct(cur.strat_CAGR);
  $("cagr-l").innerHTML = `yearly return, vs <b>${pct(cur.bh_CAGR)}</b> buy and hold`;
  $("dd").textContent = pct(cur.strat_MaxDD, 0);
  $("dd-l").innerHTML = `worst drawdown, vs <b>${pct(cur.bh_MaxDD, 0)}</b> buy and hold`;
  $("mkt").textContent = pct(cur.in_market, 0);
}

function hover(e) {
  if (!cur) return;
  const r = $("chart").getBoundingClientRect();
  const f = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
  const i = Math.round(f * (cur.dates.length - 1));
  $("cross").setAttribute("x1", f * W); $("cross").setAttribute("x2", f * W); $("cross").setAttribute("visibility", "visible");
  const tip = $("tip");
  tip.hidden = false;
  tip.innerHTML = `<b>${new Date(cur.dates[i]).toLocaleDateString(undefined, { year: "numeric", month: "short" })}</b><span>Model $${cur.strategy[i].toFixed(2)}</span><span>Buy and hold $${cur.hold[i].toFixed(2)}</span>`;
  const x = f * r.width;
  tip.style.left = `${x > r.width - 180 ? x - tip.offsetWidth - 12 : x + 12}px`;
}

$("chart").addEventListener("pointermove", hover);
$("chart").addEventListener("pointerleave", () => { $("tip").hidden = true; $("cross").setAttribute("visibility", "hidden"); });

fetch("data.json").then((r) => r.json()).then((d) => {
  DATA = d;
  $("tickers").innerHTML = Object.keys(d).map((t) => `<button data-t="${t}" aria-pressed="false">${NAMES[t] || t}</button>`).join("");
  document.querySelectorAll("#tickers button").forEach((b) => b.addEventListener("click", () => draw(b.dataset.t)));
  draw("AAPL");
});

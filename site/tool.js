// The KeynesAI Signal: asks /api/forecast (evaluation/walk_forward.py on live prices) and sets it as a news story.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const POPULAR = [["^GSPC", "S&P 500"], ["SPY", "SPY"], ["AAPL", "AAPL"], ["MSFT", "MSFT"], ["NVDA", "NVDA"], ["AMZN", "AMZN"], ["TSLA", "TSLA"], ["IBM", "IBM"], ["BTC-USD", "BTC"]];
  const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
  const signed = (v) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(0)}%`;
  const longDate = (s) => new Date(s + "T12:00:00Z").toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric", year: "numeric", timeZone: "UTC" });
  let current = null;

  $("today").textContent = new Date().toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric", year: "numeric" });

  function chart(rec) {
    const s = rec.strategy, h = rec.hold, n = s.length;
    const all = s.concat(h), lo = Math.min(...all), hi = Math.max(...all);
    const ly = (v) => 220 - ((Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo) || 1)) * 205;
    const lx = (i) => 34 + (i / (n - 1)) * 566;
    const path = (arr) => arr.map((v, i) => `${i ? "L" : "M"}${lx(i).toFixed(1)},${ly(v).toFixed(1)}`).join("");
    const ticks = [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8].filter((v) => v >= lo * 0.97 && v <= hi * 1.03);
    const years = [];
    rec.dates.forEach((d, i) => { const y = d.slice(0, 4); if (i && y !== rec.dates[i - 1].slice(0, 4)) years.push([i, y]); });
    return `<svg viewBox="0 0 600 240" role="img" aria-label="Growth of one dollar, model versus buy and hold">
      ${ticks.map((v) => `<line class="grid" x1="34" x2="600" y1="${ly(v)}" y2="${ly(v)}" /><text x="0" y="${ly(v) + 4}">$${v}</text>`).join("")}
      ${years.map(([i, y]) => `<text x="${lx(i)}" y="238" text-anchor="middle">${y}</text>`).join("")}
      <path class="h" d="${path(h)}" /><path class="s" d="${path(s)}" /></svg>`;
  }

  function render(d) {
    const r = d.record, up = d.prob_up > 0.5, p = d.prob_up;
    const lean = Math.abs(p - 0.5) < 0.02 ? "barely leans" : Math.abs(p - 0.5) < 0.06 ? "leans" : "points";
    const edge = r.accuracy - r.always_up;
    const beat = r.strat_total > r.hold_total;
    const judged = edge > 0.01 ? "a modest edge over the naive guess" : edge < -0.005 ? "worse than simply guessing “up” every day" : "no real edge over simply guessing “up” every day";
    $("out").innerHTML = `
      <p class="kicker">${esc(d.ticker)} · Next session</p>
      <h1 class="headline">Model ${lean} <span class="${up ? "up" : "down"}">${up ? "higher" : "lower"}</span> for ${esc(d.name)}</h1>
      <p class="standfirst">KeynesAI puts the chance of a higher close after ${longDate(d.as_of)} at ${pct(p, 0)}. Its record on this ticker shows ${judged}.</p>
      <div class="byline"><span>Last close ${d.last_close.toLocaleString(undefined, { maximumFractionDigits: 2 })} ${esc(d.currency)}</span><span>Retrained just now on ${esc(d.ticker)} daily prices</span></div>
      <div class="cols">
        <div>
          <div class="meter" aria-label="Probability of a higher close: ${pct(p, 0)}">
            <div class="track"><i style="left:${p * 100}%"></i></div>
            <div class="scale"><span>Lower close</span><span>Even</span><span>Higher close</span></div>
          </div>
          <div class="body">
            <p>Over the past ${(r.days / 252).toFixed(1)} years of out-of-sample tests on ${esc(d.ticker)}, the model called the next day's direction correctly ${pct(r.accuracy)} of the time. A reader who simply assumed every day would close higher was right ${pct(r.always_up)} of the time.</p>
            <p>Trading on its calls, investing on predicted up days and sitting in cash otherwise, would have turned a dollar into $${(1 + r.strat_total).toFixed(2)}. Holding throughout would have made $${(1 + r.hold_total).toFixed(2)}. ${beat ? "The model came out ahead in this stretch, though one stretch proves little." : "Buying and holding did better."}</p>
            <p>The model was invested on ${pct(r.in_market, 0)} of days. Its worst peak-to-trough fall was ${pct(Math.abs(r.strat_MaxDD), 0)}, against ${pct(Math.abs(r.hold_MaxDD), 0)} for holding.</p>
          </div>
          <figure class="chart" style="margin:22px 0 0">
            <figcaption>What $1 became since ${longDate(r.from)}</figcaption>
            <div class="key"><span><i style="background:var(--oxford)"></i>Following the model</span><span><i style="background:var(--claret)"></i>Buy and hold</span></div>
            ${chart(r)}
            <p class="src">Weekly values, log scale. Walk-forward test, retrained every 250 sessions, 5bp per trade.</p>
          </figure>
        </div>
        <div>
          <table class="facts"><caption>The record in numbers</caption>
            <tr><th>Test sessions</th><td>${r.days.toLocaleString()}</td></tr>
            <tr><th>Direction right</th><td>${pct(r.accuracy)}</td></tr>
            <tr><th>Always guessing up</th><td>${pct(r.always_up)}</td></tr>
            <tr><th>Model return</th><td>${signed(r.strat_total)}</td></tr>
            <tr><th>Buy and hold</th><td>${signed(r.hold_total)}</td></tr>
            <tr><th>Worst fall, model</th><td>${pct(r.strat_MaxDD, 0)}</td></tr>
            <tr><th>Worst fall, hold</th><td>${pct(r.hold_MaxDD, 0)}</td></tr>
          </table>
          <p class="pull">“A call is only worth as much as the record behind it.”</p>
          <p style="font-size:15px;color:var(--ink-2)">The original KeynesAI backtest claimed 65% accuracy because its models had seen the test years. Retrained honestly, most tickers land near a coin flip. <a href="/about">Read the full story</a>.</p>
        </div>
      </div>`;
  }

  async function run(ticker) {
    ticker = ticker.trim().toUpperCase();
    if (!ticker) return;
    current = ticker;
    $("ticker").value = ticker;
    document.querySelectorAll("#popular button").forEach((c) => c.setAttribute("aria-pressed", String(c.dataset.t === ticker)));
    history.replaceState(null, "", `?t=${encodeURIComponent(ticker)}`);
    $("go").disabled = true;
    $("out").innerHTML = `<p class="wait">Fetching ${esc(ticker)} prices and retraining the model<span class="dots"></span><br />Usually under ten seconds.</p>`;
    try {
      const res = await fetch(`/api/forecast?ticker=${encodeURIComponent(ticker)}`);
      const d = await res.json().catch(() => ({ error: "The server sent an unexpected reply." }));
      if (ticker !== current) return;
      if (!res.ok) throw new Error(d.error || "Something went wrong.");
      render(d);
    } catch (err) {
      if (ticker !== current) return;
      $("out").innerHTML = `<div class="err"><h2>No call for ${esc(ticker)}</h2><p>${esc(err.message)}</p></div>`;
    } finally {
      $("go").disabled = false;
    }
  }

  $("popular").innerHTML = POPULAR.map(([t, n]) => `<button type="button" data-t="${esc(t)}">${esc(n)}</button>`).join("");
  $("popular").addEventListener("click", (e) => { const b = e.target.closest("[data-t]"); if (b) run(b.dataset.t); });
  $("form").addEventListener("submit", (e) => { e.preventDefault(); run($("ticker").value); });
  run(new URLSearchParams(location.search).get("t") || "^GSPC");
})();

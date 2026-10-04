"""Live monitoring dashboard (stdlib only, no extra dependencies).

Runs in a daemon thread inside the bot process and serves a single auto-refreshing page plus a JSON API:

    GET /            dashboard UI
    GET /api/state   full JSON snapshot (engine + feed + DB)

It binds to 127.0.0.1 by default because it exposes account/trade data with no authentication.
"""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable


log = logging.getLogger(__name__)


def build_state(engine: Any, feed: Any, db: Any, backend_name: str) -> dict[str, Any]:
    state = engine.snapshot()
    today = state["now"][:10]
    state["backend"] = backend_name
    state["feed"] = feed.health()
    state["equity_curve"] = [{"ts": r[0], "equity": r[1]} for r in db.query(
        "SELECT ts, equity FROM equity WHERE ts >= ? ORDER BY ts", (today,))]
    cols = ("id", "symbol", "setup", "direction", "opened_at", "closed_at", "lots", "exit_reason",
            "gross_pnl", "charges", "net_pnl", "status")
    rows = db.query(f"SELECT {','.join(cols)} FROM trades ORDER BY opened_at DESC LIMIT 50")
    state["trades"] = [dict(zip(cols, r)) for r in rows]
    agg = db.query("SELECT COUNT(*), COALESCE(SUM(net_pnl>0),0), COALESCE(SUM(net_pnl),0), COALESCE(SUM(charges),0)"
                   " FROM trades WHERE status='CLOSED' AND opened_at >= ?", (today,))[0]
    state["today"] = {"closed": agg[0], "wins": agg[1], "net": agg[2], "charges": agg[3]}
    return state


class DashboardServer:
    def __init__(self, engine: Any, feed: Any, db: Any, backend_name: str,
                 host: str = "127.0.0.1", port: int = 8050) -> None:
        self.host, self.port = host, port
        provider: Callable[[], dict[str, Any]] = lambda: build_state(engine, feed, db, backend_name)

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:                           # noqa: N802
                path = self.path.split("?", 1)[0]
                try:
                    if path == "/":
                        self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
                    elif path == "/favicon.ico":
                        self._send(204, b"", "image/x-icon")
                    elif path == "/api/state":
                        self._send(200, json.dumps(provider(), default=str).encode("utf-8"), "application/json")
                    else:
                        self._send(404, b"not found", "text/plain")
                except Exception:                               # noqa: BLE001
                    log.exception("dashboard request failed")
                    self._send(500, b"internal error", "text/plain")

            def log_message(self, *args: Any) -> None:          # silence per-request logging
                pass

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="dashboard", daemon=True)

    def start(self) -> None:
        self._thread.start()
        log.info("Dashboard: http://%s:%d", self.host, self.port)

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Paper Trading Monitor</title>
<style>
:root{color-scheme:light;--bg:#f4f4f2;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#807f7a;--line:#e3e2dd;
--s1:#2a78d6;--good:#008300;--bad:#c93a39;--warn:#b57600;--chip:#ecebe6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#111110;--surface:#1a1a19;
--ink:#fff;--ink2:#c3c2b7;--muted:#8d8c84;--line:#2e2e2b;--s1:#3987e5;--good:#3fb950;--bad:#e66767;--warn:#d9a33a;--chip:#262624}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#111110;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#8d8c84;
--line:#2e2e2b;--s1:#3987e5;--good:#3fb950;--bad:#e66767;--warn:#d9a33a;--chip:#262624}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;padding:12px 16px;border-bottom:1px solid var(--line);background:var(--surface)}
header h1{font-size:16px;margin:0 8px 0 0}
.chip{background:var(--chip);border-radius:999px;padding:2px 10px;font-size:12px;color:var(--ink2);white-space:nowrap}
.chip.ok::before{content:"● ";color:var(--good)}.chip.bad::before{content:"▲ ";color:var(--bad)}.chip.warn::before{content:"◆ ";color:var(--warn)}
header .sp{flex:1}
button{background:var(--chip);color:var(--ink);border:0;border-radius:6px;padding:4px 10px;cursor:pointer}
main{max-width:1280px;margin:0 auto;padding:16px;display:grid;gap:16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px;min-width:0}
.card h2{font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);margin:0 0 8px;font-weight:600}
.kpi .v{font-size:24px;font-weight:650;font-variant-numeric:tabular-nums}.kpi .s{color:var(--ink2);font-size:12px}
.pos{color:var(--good)}.neg{color:var(--bad)}
.grid2{display:grid;grid-template-columns:2fr 1fr;gap:16px}@media(max-width:900px){.grid2{grid-template-columns:1fr}}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;font-weight:600}
td.n,th.n{text-align:right}tr.leg td{color:var(--ink2);font-size:12px;border-bottom:0;padding-top:0}
.empty{color:var(--muted);padding:8px 0}
.bar{height:8px;background:var(--chip);border-radius:4px;position:relative;overflow:hidden;min-width:90px}
.bar>i{position:absolute;left:0;top:0;bottom:0;background:var(--s1);border-radius:4px}
.cat{display:grid;grid-template-columns:70px 1fr 120px;gap:8px;align-items:center;margin:6px 0;font-size:13px}
svg text{fill:var(--muted);font-size:11px}
.tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:4px 8px;font-size:12px;display:none;box-shadow:0 2px 8px #0003}
ul.plain{margin:0;padding-left:18px;color:var(--ink2);font-size:13px}
.rng{position:relative;height:18px;min-width:180px}
.rng .t{position:absolute;top:8px;height:3px;background:var(--line);left:0;right:0}
.rng .b{position:absolute;top:5px;height:9px;background:var(--s1);opacity:.35;border-radius:3px}
.rng .m{position:absolute;top:2px;width:2px;height:15px;background:var(--s1)}
.rng .a{position:absolute;top:2px;width:2px;height:15px;background:var(--ink)}
</style></head><body>
<header><h1>Paper Trading Monitor</h1><span id="clock" class="chip"></span><span id="backend" class="chip"></span>
<span id="feed" class="chip"></span><span id="sessions"></span><span class="sp"></span>
<span class="chip" id="upd">connecting…</span><button id="theme" title="Toggle theme">Theme</button></header>
<main>
<section class="kpis" id="kpis"></section>
<section class="grid2">
 <div class="card"><h2>Equity today</h2><div id="curve"></div></div>
 <div class="card"><h2>Margin by category (cap 30% of equity)</h2><div id="margin"></div></div>
</section>
<section class="card"><h2>Open positions</h2><div class="scroll" id="positions"></div></section>
<section class="card"><h2>Latest forecasts (horizon end; range bar = q10–q90, blue tick = q50, black tick = last close)</h2><div class="scroll" id="preds"></div></section>
<section class="card"><h2>Recent trades</h2><div class="scroll" id="trades"></div></section>
<section class="grid2">
 <div class="card"><h2>Feed health</h2><div class="scroll" id="health"></div></div>
 <div class="card"><h2>Recent skips</h2><div id="skips"></div></div>
</section>
</main><div class="tip" id="tip"></div>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const inr=(v,d=0)=>v==null?"–":(v<0?"−":"")+"₹"+Math.abs(v).toLocaleString("en-IN",{minimumFractionDigits:d,maximumFractionDigits:d});
const sgn=(v,d=0)=>v==null?"–":(v>0?"+":v<0?"−":"")+"₹"+Math.abs(v).toLocaleString("en-IN",{minimumFractionDigits:d,maximumFractionDigits:d});
const cls=v=>v>0?"pos":v<0?"neg":"";
const num=(v,d=2)=>v==null?"–":Number(v).toLocaleString("en-IN",{minimumFractionDigits:d,maximumFractionDigits:d});
const hm=s=>s?new Date(s).toLocaleTimeString("en-IN",{hour12:false,timeZone:"Asia/Kolkata"}):"–";
const dirTxt=d=>d>0?"▲ Bull":d<0?"▼ Bear":"◆ Neutral";
const table=(cols,rows,empty)=>rows.length?`<table><thead><tr>${cols.map(c=>`<th class="${c[2]||""}">${c[0]}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table>`:`<div class="empty">${empty}</div>`;
try{const t=localStorage.getItem("theme");if(t)document.documentElement.dataset.theme=t}catch(e){}
$("theme").onclick=()=>{const d=document.documentElement,dark=getComputedStyle(d).colorScheme==="dark";d.dataset.theme=dark?"light":"dark";try{localStorage.setItem("theme",d.dataset.theme)}catch(e){}};

function curve(pts,cap){
  if(pts.length<2)return `<div class="empty">Equity snapshots appear once per minute.</div>`;
  const W=640,H=200,L=64,R=10,T=10,B=22;
  const xs=pts.map(p=>new Date(p.ts).getTime()),ys=pts.map(p=>p.equity);
  let lo=Math.min(...ys,cap),hi=Math.max(...ys,cap);if(hi-lo<1){hi+=1;lo-=1}const pad=(hi-lo)*.1;lo-=pad;hi+=pad;
  const x0=xs[0],x1=xs[xs.length-1]||x0+1,X=t=>L+(t-x0)/(x1-x0||1)*(W-L-R),Y=v=>T+(hi-v)/(hi-lo)*(H-T-B);
  window._cpx=pts.map((p,i)=>[X(xs[i]),Y(p.equity)]);
  const d=window._cpx.map((q,i)=>(i?"L":"M")+q[0].toFixed(1)+" "+q[1].toFixed(1)).join("");
  let g="";for(let i=0;i<=3;i++){const v=lo+(hi-lo)*i/3;g+=`<line x1="${L}" x2="${W-R}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/><text x="${L-6}" y="${Y(v)+4}" text-anchor="end">${inr(v)}</text>`}
  g+=`<line x1="${L}" x2="${W-R}" y1="${Y(cap)}" y2="${Y(cap)}" stroke="var(--muted)" stroke-dasharray="4 3"/><text x="${W-R}" y="${Y(cap)-4}" text-anchor="end">start capital</text>`;
  g+=`<text x="${L}" y="${H-6}">${hm(pts[0].ts)}</text><text x="${W-R}" y="${H-6}" text-anchor="end">${hm(pts[pts.length-1].ts)}</text>`;
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" id="csvg" role="img" aria-label="Equity curve">${g}<path d="${d}" fill="none" stroke="var(--s1)" stroke-width="2" stroke-linejoin="round"/><line id="ch" y1="${T}" y2="${H-B}" stroke="var(--muted)" style="display:none"/><circle id="cd" r="4" fill="var(--s1)" stroke="var(--surface)" stroke-width="2" style="display:none"/></svg>`;
}
function wireCurve(pts){
  const svg=$("csvg");if(!svg)return;const tip=$("tip"),px=window._cpx;
  svg.onmousemove=e=>{const r=svg.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*640;
    let best=0,bd=1e18;px.forEach((q,i)=>{const dd=Math.abs(q[0]-x);if(dd<bd){bd=dd;best=i}});
    const q=px[best],p=pts[best],ch=$("ch"),cd=$("cd");
    ch.setAttribute("x1",q[0]);ch.setAttribute("x2",q[0]);ch.style.display="";
    cd.setAttribute("cx",q[0]);cd.setAttribute("cy",q[1]);cd.style.display="";
    tip.style.display="block";tip.style.left=(e.clientX+12)+"px";tip.style.top=(e.clientY+12)+"px";tip.textContent=hm(p.ts)+"  "+inr(p.equity);};
  svg.onmouseleave=()=>{$("tip").style.display="none";$("ch").style.display="none";$("cd").style.display="none"};
}
function rng(p){
  const lo=Math.min(p.q10,p.anchor),hi=Math.max(p.q90,p.anchor),w=(hi-lo)||1,pc=v=>((v-lo)/w*96+2).toFixed(1)+"%";
  return `<div class="rng" title="q10 ${num(p.q10)}  q50 ${num(p.q50)}  q90 ${num(p.q90)}  close ${num(p.anchor)}"><div class="t"></div><div class="b" style="left:${pc(p.q10)};width:${((p.q90-p.q10)/w*96).toFixed(1)}%"></div><div class="m" style="left:${pc(p.q50)}"></div><div class="a" style="left:${pc(p.anchor)}"></div></div>`;
}
function render(s){
  $("clock").textContent=new Date(s.now).toLocaleString("en-IN",{timeZone:"Asia/Kolkata",hour12:false})+" IST";
  const mock=s.backend==="mock";
  $("backend").className="chip "+(mock?"warn":"ok");$("backend").textContent="model: "+s.backend+(mock?" (NOT TimesFM)":"");
  const f=s.feed,fok=f.stream_open||f.poll_only;
  $("feed").className="chip "+(fok?"ok":"bad");$("feed").textContent=f.poll_only?"feed: REST polling":("feed: "+(f.stream_open?"stream open":"stream down"));
  $("sessions").innerHTML=Object.entries(s.sessions).map(([k,v])=>`<span class="chip ${v.state==="trading"?"ok":""}">${k}: ${esc(v.state)} · sq-off ${v.square_off}</span>`).join(" ");
  const day=s.realized+s.unrealized,t=s.today;
  $("kpis").innerHTML=[
   ["Equity",inr(s.equity),`start ${inr(s.capital)}`,""],
   ["Total P&L",sgn(day),`realized ${sgn(s.realized)} · open ${sgn(s.unrealized)}`,cls(day)],
   ["Closed trades",t.closed,t.closed?`${t.wins} win · ${t.closed-t.wins} loss · fees ${inr(t.charges)}`:"none yet",""],
   ["Open positions",s.positions.length,`risk/trade cap ${inr(s.risk_per_trade)}`,""]
  ].map(k=>`<div class="card kpi"><h2>${k[0]}</h2><div class="v ${k[3]}">${k[1]}</div><div class="s">${k[2]}</div></div>`).join("");
  $("curve").innerHTML=curve(s.equity_curve,s.capital);wireCurve(s.equity_curve);
  $("margin").innerHTML=Object.entries(s.margin_used).map(([c,u])=>{const p=Math.min(100,u/s.margin_cap*100);
    return `<div class="cat"><span>${c}</span><div class="bar"><i style="width:${p}%"></i></div><span class="n">${inr(u)} / ${inr(s.margin_cap)}</span></div>`}).join("");
  const prow=[];s.positions.forEach(p=>{
    prow.push(`<tr><td><b>${esc(p.symbol)}</b></td><td>${esc(p.setup)}</td><td>${dirTxt(p.direction)}</td><td class="n">${p.lots}</td><td>${hm(p.opened_at)}</td>
    <td class="n">${num(p.entry_underlying)}</td><td class="n">${num(p.stop)}</td><td class="n">${num(p.target)}</td>
    <td class="n ${cls(p.pnl)}"><b>${sgn(p.pnl)}</b></td><td class="n">${inr(p.risk_limit)}</td><td class="n">${p.pnl_floor==null?"–":sgn(p.pnl_floor)}</td></tr>`);
    p.legs.forEach(l=>prow.push(`<tr class="leg"><td colspan="3">&nbsp;&nbsp;${l.side>0?"BUY":"SELL"} ${esc(l.symbol)}</td><td class="n">${l.qty}</td><td></td><td class="n">in ${num(l.entry)}</td><td class="n">ltp ${num(l.last)}</td><td></td><td class="n ${cls(l.pnl)}">${sgn(l.pnl)}</td><td></td><td></td></tr>`))});
  $("positions").innerHTML=table([["Symbol"],["Setup"],["Bias"],["Lots","n"],["Opened"],["Entry px","n"],["Stop","n"],["Target","n"],["P&L","n"],["Risk cap","n"],["Trail floor","n"]],prow,"No open positions.");
  $("preds").innerHTML=table([["Symbol"],["Bar"],["Close","n"],["q10","n"],["q50","n"],["q90","n"],["Drift","n"],["Envelope","n"],["ATR20","n"],["Range"],["Latency","n"]],
    s.predictions.map(p=>`<tr><td><b>${esc(p.symbol)}</b>${p.expiry_day?' <span class="chip">expiry</span>':""}</td><td>${hm(p.bar_start)}</td><td class="n">${num(p.anchor)}</td><td class="n">${num(p.q10)}</td><td class="n">${num(p.q50)}</td><td class="n">${num(p.q90)}</td>
    <td class="n ${cls(p.drift)}">${p.drift>0?"▲ +":p.drift<0?"▼ −":""}${num(Math.abs(p.drift))}</td><td class="n">${num(p.spread)}</td><td class="n">${num(p.atr20)}</td><td>${rng(p)}</td><td class="n">${num(p.latency_s,2)}s</td></tr>`),"Waiting for the first candle close…");
  $("trades").innerHTML=table([["Symbol"],["Setup"],["Bias"],["Opened"],["Closed"],["Lots","n"],["Exit reason"],["Gross","n"],["Fees","n"],["Net","n"],["Status"]],
    s.trades.map(t=>`<tr><td><b>${esc(t.symbol)}</b></td><td>${esc(t.setup)}</td><td>${dirTxt(t.direction)}</td><td>${hm(t.opened_at)}</td><td>${hm(t.closed_at)}</td><td class="n">${t.lots}</td><td>${esc(t.exit_reason||"")}</td>
    <td class="n ${cls(t.gross_pnl)}">${t.gross_pnl==null?"–":sgn(t.gross_pnl)}</td><td class="n">${t.charges==null?"–":inr(t.charges)}</td><td class="n ${cls(t.net_pnl)}"><b>${t.net_pnl==null?"–":sgn(t.net_pnl)}</b></td><td>${esc(t.status)}</td></tr>`),"No trades yet.");
  $("health").innerHTML=table([["Symbol"],["Last price","n"],["Tick age","n"],["5m candles","n"]],
    f.symbols.map(h=>{const stale=h.tick_age_s!=null&&h.tick_age_s>90;return `<tr><td><b>${esc(h.symbol)}</b></td><td class="n">${num(h.price)}</td><td class="n ${stale?"neg":""}">${h.tick_age_s==null?"no ticks":(stale?"▲ ":"")+Math.round(h.tick_age_s)+"s"}</td><td class="n">${h.candles}</td></tr>`}),"–");
  $("skips").innerHTML=s.rejections.length?`<ul class="plain">${s.rejections.slice().reverse().map(r=>`<li>${esc(r)}</li>`).join("")}</ul>`:`<div class="empty">No skipped signals.</div>`;
}
async function tick(){
  try{const r=await fetch("/api/state",{cache:"no-store"});if(!r.ok)throw new Error(r.status);render(await r.json());
    $("upd").className="chip ok";$("upd").textContent="live · "+new Date().toLocaleTimeString("en-IN",{hour12:false})}
  catch(e){$("upd").className="chip bad";$("upd").textContent="disconnected"}
}
tick();setInterval(tick,3000);
</script></body></html>
"""

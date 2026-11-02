// Bird Register: shared by index.html and birds.html.
(function(){
  const $ = s => document.querySelector(s);
  const bust = () => "?t=" + Math.floor(Date.now()/30000);
  const j = async p => { const r = await fetch("data/"+p+bust(), {cache:"no-store"}); if(!r.ok) throw new Error(p); return r.json(); };
  const fmtT = iso => new Date(iso).toLocaleString("en-AU",{weekday:"short",day:"numeric",month:"short",hour:"2-digit",minute:"2-digit",timeZone:"Australia/Sydney"});
  const fmtD = iso => new Date(iso).toLocaleDateString("en-AU",{weekday:"long",day:"numeric",month:"long",timeZone:"Australia/Sydney"});
  const sp = t => (t||"").replace(/\s*\(guess\)/g,"");
  const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
  const hasWindow = !!$("#window");

  let state=null, events=[], birds=[], motion=0, lum=0.5;
  let box = {list:[], i:0, title:""};

  function describe(e, long){
    const what = e.kind==="flyer" ? "flew" + (e.direction ? " " + e.direction : " past") : "sat there for " + e.duration + " seconds";
    const who = e.confidence==="likely" ? "probably the same bird as before, visit " + e.visit : "first time we have seen this one";
    return long ? `${fmtT(e.time)}<br>${what}<br>${sp(e.species)}, ${e.size}, ${e.tone}<br>${who}<br>sky was ${e.sky||"unrecorded"}`
                : `${what}, ${e.size}, ${e.tone}. ${sp(e.species)}`;
  }

  // ---------- lightbox ----------
  function openBox(list, i, title){
    if(!list.length) return;
    box = {list, i, title}; $("#box").classList.add("open"); showBox();
  }
  function showBox(){
    const e = box.list[box.i];
    $("#boximg").src = "data/" + e.image + bust();
    $("#boxcap").innerHTML = `<b>${esc(e.bird)}</b>${e.visit>1?" · visit "+e.visit:""}<br>${describe(e,true)}`;
    $("#boxcount").textContent = `${box.title ? box.title + " · " : ""}${box.i+1} of ${box.list.length}`;
    $("#boxprev").style.visibility = box.i < box.list.length-1 ? "visible" : "hidden";
    $("#boxnext").style.visibility = box.i > 0 ? "visible" : "hidden";
  }
  function closeBox(){ $("#box").classList.remove("open"); }
  $("#box").addEventListener("click", ev => { if(ev.target.id==="box" || ev.target.id==="boxclose") closeBox(); });
  $("#boxprev").addEventListener("click", () => { if(box.i < box.list.length-1){ box.i++; showBox(); } });   // older
  $("#boxnext").addEventListener("click", () => { if(box.i > 0){ box.i--; showBox(); } });                  // newer
  document.addEventListener("keydown", ev => {
    if(!$("#box").classList.contains("open")) return;
    if(ev.key==="Escape") closeBox();
    if(ev.key==="ArrowLeft"){ if(box.i < box.list.length-1){ box.i++; showBox(); } }
    if(ev.key==="ArrowRight"){ if(box.i > 0){ box.i--; showBox(); } }
  });

  function visitsOf(b){ return events.filter(e => e.bird_id===b.id); }

  // ---------- render ----------
  function render(){
    // latest
    if(events.length){
      const e = events[0];
      $("#birdimg").src = "data/"+e.image+bust();
      $("#birdname").textContent = e.bird;
      $("#birdmeta").innerHTML = describe(e, true);
      $("#birdimg").onclick = () => openBox(events, 0, "all entries");
      $("#latestwrap").classList.remove("empty");
    }
    // catalogue
    const sorted = birds.slice().sort((a,b)=>new Date(b.last_seen)-new Date(a.last_seen));
    $("#catcount").textContent = sorted.length ? `${sorted.length} named` : "";
    $("#gallery").innerHTML = sorted.map(b => {
      const v = visitsOf(b); const main = v[0];
      return `<div class="gb" data-bird="${b.id}">${main?`<img src="data/${main.image}${bust()}" alt="">`:""}<b>${esc(b.name)}</b>${b.visits} visit${b.visits===1?"":"s"} · ${b.kind}<br>${sp(b.species||"")}<br><span class="when">first ${fmtT(b.first_seen)}</span>${v.length>1?`<div class="more">${v.slice(1,7).map((e,k)=>`<img src="data/${e.image}${bust()}" alt="" data-bird="${b.id}" data-k="${k+1}">`).join("")}</div>`:""}</div>`;
    }).join("");
    $("#gallery").querySelectorAll(".gb").forEach(card => {
      card.addEventListener("click", ev => {
        const b = birds.find(x => x.id===card.dataset.bird); const v = visitsOf(b);
        const k = ev.target.dataset.k ? parseInt(ev.target.dataset.k,10) : 0;
        openBox(v, Math.min(k, v.length-1), "every visit by " + b.name);
      });
    });
    // ledger grouped by day
    $("#ledgercount").textContent = events.length ? `${events.length} entries` : "";
    let html = "", day = "";
    events.forEach((e, i) => {
      const d = fmtD(e.time);
      if(d !== day){ day = d; html += `<tr class="day"><td colspan="4">${d}</td></tr>`; }
      html += `<tr class="${e.kind}" data-i="${i}"><td class="i"><img src="data/${e.image}${bust()}" alt="" loading="lazy"></td><td class="t">${new Date(e.time).toLocaleTimeString("en-AU",{hour:"2-digit",minute:"2-digit",timeZone:"Australia/Sydney"})}</td><td><b>${esc(e.bird)}</b>${e.visit>1?" ×"+e.visit:""}</td><td class="s">${describe(e,false)}</td></tr>`;
    });
    $("#ledger").innerHTML = html;
    $("#ledger").querySelectorAll("tr[data-i]").forEach(tr => tr.addEventListener("click", () => openBox(events, parseInt(tr.dataset.i,10), "all entries")));
  }

  async function loadState(){
    try{
      state = await j("state.json");
      const age = (Date.now() - new Date(state.time).getTime())/1000;
      const dot = $("#dot"); if(dot) dot.classList.toggle("stale", age > 900);
      document.documentElement.style.setProperty("--sky", state.sky.hex);
      $("#livetxt").innerHTML = age > 900 ? `last heard <b>${fmtT(state.time)}</b>` : `sky <b>${state.sky.word}</b> · birds today <b>${state.birds_today}</b> · named so far <b>${state.named_birds}</b>`;
      const up = $("#uptime"); if(up) up.textContent = ` The iMac has been watching since ${fmtT(state.started)}.`;
      motion = state.motion; lum = state.overall_lum;
      if(hasWindow) $("#viewtime").textContent = "Taken " + fmtT(state.time) + ". " + (state.sky.word ? "Sky " + state.sky.word + ", " : "") + "light " + Math.round(state.overall_lum*100) + "%.";
    }catch(e){ $("#livetxt").textContent = "nothing from the iMac yet"; }
  }
  async function loadBirds(){
    try{
      const [led, cat] = await Promise.all([j("ledger.json"), j("birds.json")]);
      events = led.events || []; birds = cat.birds || []; render();
    }catch(e){}
  }

  // ---------- the window (index.html only) ----------
  let edgePts = [], showLong = false, showEdges = true;
  function loadEdges(){
    const im = new Image();
    im.onload = () => {
      const o = document.createElement("canvas"); o.width = im.width; o.height = im.height;
      const c = o.getContext("2d"); c.drawImage(im,0,0);
      const d = c.getImageData(0,0,o.width,o.height).data; const pts=[];
      for(let y=0;y<o.height;y+=2) for(let x=0;x<o.width;x+=2){ if(d[(y*o.width+x)*4] > 128) pts.push([x/o.width, y/o.height]); }
      edgePts = pts;
    };
    im.src = "data/edges.png"+bust();
  }
  function loadView(){
    $("#lx").src = "data/" + (showLong ? "longexposure.jpg" : "latest.jpg") + bust();
    $("#viewtag").textContent = showLong ? "every frame since the start, blended into one picture" : "the view a few minutes ago";
  }
  if(hasWindow){
    const cv = $("#geom"), ctx = cv.getContext("2d"); const t0 = performance.now();
    function draw(now){
      const w = cv.clientWidth, h = cv.clientHeight;
      if(cv.width!==w||cv.height!==h){cv.width=w;cv.height=h;}
      ctx.clearRect(0,0,w,h);
      if(showEdges){
        const t = (now - t0)/1000, shake = 0.6 + motion*60, a = (0.45 + lum*0.45).toFixed(2);
        ctx.fillStyle = `rgba(255,255,255,${a})`; ctx.shadowColor = "rgba(61,48,64,.9)"; ctx.shadowBlur = 2;
        const n = edgePts.length, step = n > 6000 ? Math.ceil(n/6000) : 1;
        for(let i=0;i<n;i+=step){ const p = edgePts[i];
          ctx.fillRect(p[0]*w + Math.sin(t*1.7 + p[1]*40 + i*0.01)*shake, p[1]*h + Math.cos(t*1.3 + p[0]*30)*shake*0.6, 2, 2); }
      }
      requestAnimationFrame(draw);
    }
    requestAnimationFrame(draw);
    $("#lxtoggle").addEventListener("click", () => { showLong = !showLong; loadView(); $("#lxtoggle").textContent = showLong ? "show the view" : "show long exposure"; });
    $("#edgetoggle").addEventListener("click", () => { showEdges = !showEdges; $("#edgetoggle").textContent = showEdges ? "hide the lines" : "show the lines"; });
    $("#lx").addEventListener("click", () => { $("#bigview").src = $("#lx").src; $("#viewbox").classList.add("open"); });
    $("#viewbox").addEventListener("click", () => $("#viewbox").classList.remove("open"));
  }

  async function refresh(){ await loadState(); await loadBirds(); if(hasWindow){ loadView(); loadEdges(); } }
  refresh(); setInterval(refresh, 60000);
})();

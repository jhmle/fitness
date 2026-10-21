import { PROFILE, VAD_URL, ORT_URL, LLM_ID, KOKORO_STRICT } from "./config.js";
import { loadPose, detect } from "./pose.js";
import { RepEngine, SKELETON } from "./exercises.js";
import { Agent, kbDocs } from "./agent.js";

const $ = (id) => document.getElementById(id);
const setStatus = (text, kind = "idle") => { const s = $("status"); s.textContent = text; s.className = "pill pill-" + kind; };

const moon = new Worker("moonshine.worker.js", { type: "module" });
const kokoro = PROFILE.tts === "kokoro" ? new Worker("kokoro.worker.js", { type: "module" }) : null;
const llmW = PROFILE.llm ? new Worker("llm.worker.js?v=4", { type: "module" }) : null;
const need = 2 + (kokoro ? 1 : 0) + (llmW ? 1 : 0);                       // pose+VAD, Moonshine, (Kokoro)
let llmModel = null, ready = 0, starting = false, S = null, state = "off", vad = null, paused = false, busyTTS = 0, epoch = 0;
let lastMeter = 0;
let micStream = null;                                                       // mic stream we open ourselves and hand to the VAD
for (const [w, name] of [[moon, "Moonshine"], [kokoro, "Kokoro"], [llmW, "Coach brain"]])   // a worker that dies (often out of memory) used to fail silently
  if (w) w.onerror = (ev) => {
    ev.preventDefault?.(); fail(new Error(ev.message || "worker stopped unexpectedly"), name); memCheck(name, ev.message || "", !ev.message);
    if (w === kokoro) kokoroDown(ev.message || "worker stopped unexpectedly");
  };
const engine = new RepEngine("squat");
const PTS = [...new Set(SKELETON.flat())];

if (!llmW) { $("lLabel").hidden = true; $("policy").parentElement.hidden = true; $("modelNote").hidden = true; }
if (!kokoro) { $("kLabel").hidden = true; $("voice").parentElement.hidden = true; $("fastWrap").hidden = true; }

// ---------------------------------------------------------------- states (what the stage shows)
const STATES = {
  off:       ["Not started", "Ready when you are", "Start a session and set the camera so your whole body is in view."],
  listening: ["Tracking", "Go ahead, I'm watching", "Stand side-on to the camera. Say a command any time."],
  hearing:   ["Voice", "Hearing you…", "Pause when you're done."],
  thinking:  ["Coach", "Thinking…", "Working out a reply."],
  speaking:  ["Coach", "Speaking", "Press Skip cue to cut it short."],
};
function setState(s) {
  state = s;
  const [turn, title, hint] = STATES[s];
  $("stage").dataset.state = s;
  $("turnText").textContent = turn; $("title").textContent = title; $("hint").textContent = hint;
  if (s === "off") { if (ready === need) setStatus("Ready", "ready"); } else setStatus("Live · " + turn.toLowerCase(), s === "listening" || s === "hearing" ? "ready" : "busy");
  refresh();
}
function refresh() {
  $("mic").disabled = ready < need || starting;
  $("skip").disabled = !S || !(state === "thinking" || state === "speaking");
}
const hud = () => { $("repN").textContent = agent.reps.length + "/" + agent.target; $("setN").textContent = agent.set; };

// ---------------------------------------------------------------- loading
// Low-memory detection. WASM/ONNX out-of-memory shows up as allocation errors, bare numeric pointer codes, or a model that just stops answering.
const looksLikeMemory = (m) => /out of memory|memory|alloc|\boom\b|bad_alloc/i.test(m) || /^\d{5,}$/.test(String(m).trim());
function memCheck(who, msg = "", soft = false) {
  const hard = looksLikeMemory(msg);
  if (!hard && !soft) return;
  const tail = "Switch to a more powerful device with more RAM (a recent laptop or desktop), or close other tabs and apps and reload. The coach still works with Fast voice and the Rules only coach.";
  note("memNote", hard ? `Not enough memory: this device can't hold the on-device AI models (${who}). ${tail}`
                       : `${who} stopped responding, which often means the device is running out of memory. If it keeps happening: ${tail[0].toLowerCase() + tail.slice(1)}`);
}
const fail = (e, who = "") => {                      // always say which component failed
  const msg = String(e.message || e); console.error(who, e); memCheck(who || "A model", msg);
  $("diag").textContent = who + ": " + msg;
  setStatus(`Error${who ? " (" + who + ")" : ""}: ` + msg.slice(0, 180), "error");
};
const onReady = () => { if (++ready < need) return; $("setup").classList.add("done"); setState("off"); };
function loadScript(src) {
  return new Promise((res, rej) => { const s = document.createElement("script"); s.src = src; s.onload = res; s.onerror = () => rej(new Error("failed to load " + src)); document.head.append(s); });
}
$("load").onclick = async () => {
  $("load").disabled = true; setStatus("Downloading models…", "loading");
  moon.postMessage({ type: "load" });
  if (kokoro) { kokoro.postMessage({ type: "load", strict: KOKORO_STRICT }); kBump(); }
  if (llmW) { llmW.postMessage({ type: "load", kb: kbDocs() }); }
  try {
    const info = await loadPose(); $("pp").value = 60; $("diag").textContent = "pose: " + info;
    try { await loadScript(ORT_URL + "ort.wasm.min.js"); await loadScript(VAD_URL + "bundle.min.js"); }
    catch (e) { return fail(e, "Voice detection"); }
    $("pp").value = 100; onReady();
  } catch (e) { fail(e, "Pose"); }
};
moon.onmessage = async ({ data: d }) => {
  if (d.type === "progress") $("pm").value = d.pct;
  else if (d.type === "ready") { $("pm").value = 100; onReady(); }
  else if (d.type === "text") {
    $("diag").textContent = "heard: " + d.text;
    const handled = d.text && await utter(d.text);
    if (d.text && !handled) $("cueOut").textContent = "Heard: " + d.text;
    hud(); if (!busyTTS && !chatBusy && S) setState("listening");
  }
  else if (d.type === "error") { fail(d, "Moonshine"); if (S) setState("listening"); }
};

// ---------------------------------------------------------------- speech out (Kokoro on desktop, browser voice on mobile / as fallback)
const AC = window.AudioContext || window.webkitAudioContext;
let actx, analyser, nextT = 0, sources = [], lastSay = "", kT = 0, fbEpoch = -1;
let kokoroBad = false, kLast = 0, kWarm = 0, kMiss = 0, speakingWith = "kokoro", kWarming = false, kReady = false, kCounted = false, kLoadT = 0, kErr = 0;
const utts = new Set();                          // keep references: Safari can garbage-collect an utterance before it finishes
// Persistent, visible messages. (The status pill is overwritten on every state change, so errors used to vanish.)
function note(id, text) { const el = $(id); el.hidden = !text; (el.querySelector("span") || el).textContent = text || ""; if (text) $("diag").textContent = text; }

// Runs inside the Start click: unlocks Web Audio AND speechSynthesis (Safari blocks both until a user gesture, silently).
function unlockAudio() {
  try {
    actx ??= new AC(); actx.resume().catch(() => {});
    const s = actx.createBufferSource(); s.buffer = actx.createBuffer(1, 1, 22050); s.connect(actx.destination); s.start(0);
  } catch (e) { note("audioNote", "Audio output unavailable: " + (e.message || e)); }
  try { const u = new SpeechSynthesisUtterance("."); u.volume = 0; speechSynthesis.speak(u); } catch {}
}
function pickVoice() {
  const vs = (speechSynthesis.getVoices && speechSynthesis.getVoices()) || [];
  return vs.find((v) => /^en/i.test(v.lang) && v.default) || vs.find((v) => /^en/i.test(v.lang)) || vs[0] || null;
}
function browserSpeak(text, e = epoch) {
  setTimeout(() => {                             // Safari drops a speak() issued right after cancel()
    if (e !== epoch) return;
    try {
      speechSynthesis.resume();
      const u = new SpeechSynthesisUtterance(text), v = pickVoice();
      if (v) { u.voice = v; u.lang = v.lang; } else u.lang = "en-US";
      const fin = () => { utts.delete(u); if (e === epoch) done(); };
      u.onend = fin;
      u.onerror = (ev) => { if (ev.error && !/cancel|interrupt/i.test(ev.error)) note("audioNote", "Browser voice failed: " + ev.error); fin(); };
      utts.add(u); speechSynthesis.speak(u);
    } catch (x) { note("audioNote", "Browser voice failed: " + (x.message || x)); if (e === epoch) done(); }
  }, 60);
}
async function play(samples, rate, e) {
  try {
    actx ??= new AC();
    if (actx.state !== "running") await Promise.race([actx.resume(), new Promise((r) => setTimeout(r, 1500))]);   // Safari can leave resume() pending
    if (actx.state !== "running") throw new Error("the browser is blocking audio (" + actx.state + ")");
    if (e !== epoch) return;
    if (!analyser) { analyser = actx.createAnalyser(); analyser.fftSize = 256; analyser.smoothingTimeConstant = 0.6; analyser.connect(actx.destination); }
    const b = actx.createBuffer(1, samples.length, rate); b.copyToChannel(samples, 0);
    const s = actx.createBufferSource(); s.buffer = b; s.connect(analyser);      // via the analyser so the UI can draw the waves
    nextT = Math.max(nextT, actx.currentTime + 0.05); s.start(nextT); nextT += b.duration;
    sources.push(s);
    s.onended = () => { sources = sources.filter((x) => x !== s); done(); };
  } catch (x) {
    note("audioNote", "Couldn't play Kokoro audio (" + (x.message || x) + "). Using the browser voice instead.");
    browserSpeak(lastSay, e);
  }
}
function stopSpeech() {
  epoch++; busyTTS = 0; nextT = 0; clearTimeout(kT);
  sources.forEach((s) => { s.onended = null; try { s.stop(); } catch {} }); sources = [];
  utts.clear(); try { speechSynthesis.cancel(); } catch {}
}
function done() {
  busyTTS = Math.max(0, busyTTS - 1);
  if (!busyTTS && S) setTimeout(() => { if (!busyTTS && S) setState("listening"); }, 300);   // gap so the mic doesn't catch our own tail
}
function say(text, { urgent = false } = {}) {
  $("cueOut").textContent = text;
  if (!S || !$("speakOn").checked) return;
  if (busyTTS && !urgent) return;                  // don't talk over ourselves: drop routine cues
  if (urgent) stopSpeech();
  lastSay = text; busyTTS++; setState("speaking");
  const e = epoch;
  setTimeout(() => { if (e === epoch && busyTTS && S) { busyTTS = 0; setState("listening"); } }, 20000);   // never stay deaf if audio fails to finish
  const fast = useBrowserVoice();
  speakingWith = fast ? "browser" : "kokoro";
  if (!fast) {
    kokoro.postMessage({ type: "speak", text, voice: $("voice").value, id: e });
    clearTimeout(kT);                              // limit scales with how long Kokoro really takes on this device
    const limit = Math.max(6000, (kLast || kWarm || 2000) * 4);   // slow reply: browser voice takes over
    kT = setTimeout(() => {
      if (!KOKORO_STRICT && e === epoch && busyTTS && !sources.length) {
        fbEpoch = e;
        if (++kMiss >= 2) { kokoroBad = true; syncVoiceUI(); note("audioNote", "Kokoro is too slow on this device, so I switched to the browser voice for this session."); memCheck("Kokoro", "", true); }
        else note("audioNote", `Kokoro took longer than ${Math.round(limit / 1000)}s, using the browser voice for this reply.`);
        browserSpeak(text, e);
      }
    }, limit);
  } else browserSpeak(text, e);
}
function useBrowserVoice() { return !kokoro || $("fastVoice").checked || kokoroBad || (!KOKORO_STRICT && kWarming); }
function syncVoiceUI() {
  const fast = useBrowserVoice();
  $("voice").disabled = fast;
  $("voiceEngine").textContent = "Speaking with: " + (kWarming && !$("fastVoice").checked && !kokoroBad ? "browser voice while Kokoro warms up…" : fast ? "browser voice (fast)" : "Kokoro" + (kLast ? ` (about ${(kLast / 1000).toFixed(1)}s to generate a reply)` : KOKORO_STRICT && !kWarm ? " (first sentence still warming up, early replies may be delayed)" : ""));
}
// Safety net: if Kokoro can't load (error, crashed worker, or no progress for 3 minutes) or dies later, switch to the browser voice and let the app carry on.
function kBump() { clearTimeout(kLoadT); if (!kReady && !kCounted) kLoadT = setTimeout(() => kokoroDown("no progress for 30 seconds while loading"), 30000); }
function kokoroDown(why, what = kReady ? "stopped working" : "couldn't load") {
  clearTimeout(kT); clearTimeout(kLoadT);
  kokoroBad = true; kWarming = false;
  note("audioNote", `Kokoro ${what} (${why}). Using the browser voice instead.`);
  memCheck("Kokoro", why, /no progress|stopped unexpectedly/.test(why));
  if (!kCounted) { kCounted = true; $("pk").value = 100; onReady(); }      // never leave the page stuck on "Downloading models"
  syncVoiceUI();
}
kokoro && (kokoro.onmessage = ({ data: d }) => {
  if (d.type === "progress") { $("pk").value = d.pct; kBump(); }
  else if (d.type === "log") { console.log(d.message); $("diag").textContent = d.message; kBump(); }
  else if (d.type === "ready") {
    kReady = true; clearTimeout(kLoadT);
    $("voice").innerHTML = d.voices.map((v) => `<option>${v}</option>`).join("");
    $("voice").value = d.voices.includes("af_heart") ? "af_heart" : d.voices[0];
    if (kokoroBad) { kokoroBad = false; note("audioNote", ""); }          // it loaded after all (slow connection): use it
    kWarming = !KOKORO_STRICT;                                            // non-Safari: browser voice until the warm-up finishes
    syncVoiceUI(); if (!kCounted) { kCounted = true; onReady(); }
  }
  else if (d.type === "warm") {
    kWarming = false;
    if (d.bad) kokoroDown(d.why, "isn't responding in this browser");
    else { kWarm = d.ms; $("diag").textContent = `Kokoro ready. First sentence took ${(d.ms / 1000).toFixed(1)}s.`; syncVoiceUI(); }
  }
  else if (d.type === "audio") { clearTimeout(kT); kMiss = 0; kErr = 0; kLast = d.ms || kLast; $("diag").textContent = `Kokoro generated a reply in ${((d.ms || 0) / 1000).toFixed(1)}s`; syncVoiceUI(); if (d.id === epoch && d.id !== fbEpoch) play(d.samples, d.rate, d.id); }
  else if (d.type === "error") {
    clearTimeout(kT); console.error("Kokoro", d.message);
    if (!kReady) return kokoroDown(d.message);                            // load failed
    const mine = d.id === epoch && d.id !== fbEpoch;
    note("audioNote", "Kokoro voice failed: " + d.message + (mine ? ". Using the browser voice instead." : ""));
    if (mine) { if (lastSay) browserSpeak(lastSay, d.id); else done(); }
    fail(d, "Kokoro");
    if (++kErr >= 3) kokoroDown("3 failed replies in a row");
  }
});
$("skip").onclick = () => { stopSpeech(); if (S) setState("listening"); };
try { $("fastVoice").checked = localStorage.getItem("trainer.fastVoice") === "1"; } catch {}
$("fastVoice").onchange = () => {
  try { localStorage.setItem("trainer.fastVoice", $("fastVoice").checked ? "1" : "0"); } catch {}
  stopSpeech(); if (S) setState("listening"); syncVoiceUI();
};
syncVoiceUI();

// ---------------------------------------------------------------- voice waves (while the coach speaks, and while it hears you)
const wave = $("wave"), wg = wave.getContext("2d");
const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;
let micLevel = 0;
function drawWave(t) {
  requestAnimationFrame(drawWave);
  if (state !== "speaking" && state !== "hearing") return;
  const W = wave.width, H = wave.height, N = 37, step = W / N, bw = step * 0.55, mid = (N - 1) / 2, lv = [];
  let base = state === "hearing" ? micLevel : 0.55;      // the browser voice gives no audio to analyse, so it gets a simulated wave
  if (state === "speaking" && speakingWith === "kokoro") {
    if (analyser && sources.length) {                    // real audio: mirrored spectrum, loudest bins in the middle
      const d = new Uint8Array(analyser.frequencyBinCount); analyser.getByteFrequencyData(d);
      const top = Math.floor(d.length * 0.6);
      for (let i = 0; i < N; i++) lv.push(Math.min(1, d[Math.floor(Math.abs(i - mid) / mid * (top - 1))] / 190));
    } else base = 0.08;                                  // waiting for audio
  }
  if (!lv.length) for (let i = 0; i < N; i++) lv.push(calm ? base * 0.5 : base * (0.3 + 0.7 * Math.abs(Math.sin(t / 170 + i * 0.6) * Math.sin(t / 290 - i * 0.35))));
  wg.clearRect(0, 0, W, H); wg.fillStyle = "#fff";
  lv.forEach((v, i) => {
    const h = Math.max(6, v * H), x = i * step + (step - bw) / 2, y = (H - h) / 2;
    wg.beginPath(); if (wg.roundRect) wg.roundRect(x, y, bw, h, bw / 2); else wg.rect(x, y, bw, h); wg.fill();
  });
}
requestAnimationFrame(drawWave);

// ---------------------------------------------------------------- coach brain worker (LLM + embeddings)
const pend = new Map(); let rid = 0, llmOK = false, embOK = false, llmCounted = false;
const llmDone = () => { if (!llmCounted) { llmCounted = true; onReady(); } };
const call = (msg) => new Promise((res) => { const id = ++rid; pend.set(id, res); llmW.postMessage({ ...msg, id }); });
const llm = llmW && {
  ready: () => llmOK,
  chat: async (messages, max) => (await call({ type: "chat", messages, max })).text,
  chatRaw: async (messages, max) => { const r = await call({ type: "chat", messages, max }); if (r.error) throw new Error(r.message); llmRaw = r.raw || ""; return r.text; },
  search: async (query, ex, k) => (embOK ? (await call({ type: "search", query, ex, k })).ids : []),
};
if (llmW) llmW.onmessage = ({ data: d }) => {
  if (d.type === "progress") $("pl").value = d.pct;
  else if (d.type === "warn") { console.warn(d.message); $("diag").textContent = d.message; memCheck("Coach brain", d.message); }
  else if (d.type === "ready") {
    if (!d.llm) memCheck("Coach brain", d.why || ""); llmOK = d.llm; embOK = d.emb; llmModel = d.model; $("pl").value = 100;
    $("modelNote").textContent = d.llm ? "Coach brain: " + d.model + " [" + d.run + "]" : "Coach brain: NOT loaded, rules only. " + (d.why || "");
    llmDone();
  }
  else if (d.type === "result") { pend.get(d.id)?.(d); pend.delete(d.id); }
  else if (d.type === "error") { fail(d, "Coach brain"); llmDone(); }
};

// ---------------------------------------------------------------- general chat fallback
// The agent only knows workout commands + a small tips knowledge base. When it has no tip, ask the LLM instead.
let lastHeard = "", lastHeardAt = 0, chatBusy = false, llmRaw = "";
const chatHist = [];
const CHAT_SYS = "You are a friendly voice assistant and fitness coach. Answer in ONE short spoken sentence, 20 words at most. Never more than two sentences. " +
  "Plain words only: no markdown, lists, emojis or stage directions. If you are not sure, say so briefly.";
async function chatAnswer(q) {
  if (!llm || !llm.ready()) { say("My language model didn't load, so I can only handle workout commands."); return; }
  chatBusy = true; setState("thinking");
  let reply = "", err = "";
  const clean = (x) => (x || "").replace(/[*_#`>~]+/g, "").replace(/\s+/g, " ").trim();
  try {
    let t = clean(await llm.chatRaw([{ role: "system", content: CHAT_SYS }, ...chatHist.slice(-4), { role: "user", content: q }], 60));
    if (!t) t = clean(await llm.chatRaw([{ role: "user", content: q + " (Answer in one short sentence.)" }], 80));   // retry plain: no system prompt, no history
    if (!t) throw new Error("the model returned an empty answer twice. Raw output: " + llmRaw);
    const sents = t.match(/[^.!?]+[.!?]+/g);                     // keep at most two whole sentences, max ~35 words
    if (sents) t = sents.slice(0, 2).join(" ").trim(); else t += ".";
    const w = t.split(" "); if (w.length > 35) t = w.slice(0, 35).join(" ").replace(/[,;:]$/, "") + ".";
    chatHist.push({ role: "user", content: q }, { role: "assistant", content: t });   // only real answers go into history
    reply = t;
  } catch (e) {
    $("diag").textContent = "Chat error: " + (e.message || e); console.error(e);
    err = String(e.message || e).slice(0, 160);
  }
  chatBusy = false;
  if (reply) { try { say(reply, { urgent: true }); } catch (e) { console.error(e); $("cueOut").textContent = reply; } }
  else { say("Sorry, I hit a problem answering that."); $("cueOut").textContent = "Sorry, I hit a problem answering that.\n(" + err + ")"; }
  if (!busyTTS && S) setState("listening");
}
// Only workout controls go to the rule-based agent; everything else is a free-form question for the LLM.
const CMD = /\b(start|begin|switch to|change to|stop|pause|resume|rest|break|reps?|sets?|easier|harder|longer|shorter|next set|how was|how did|how many|count)\b/i;
function utter(t) { lastHeard = t; lastHeardAt = performance.now(); if (CMD.test(t) || !llm?.ready()) return agent.onVoice(t); chatAnswer(t); return true; }
const route = (text, o) => (lastHeard && llm?.ready() && /don'?t have a tip/i.test(text) ? chatAnswer(lastHeard) : say(text, performance.now() - lastHeardAt < 20000 ? { ...o, urgent: true } : o));   // replies to what you said are never dropped

// ---------------------------------------------------------------- agent
const agent = new Agent({
  say: route, llm, hud: () => hud(), policy: () => $("policy").value, stats: (t) => ($("stats").textContent = t),
  startExercise(name) { engine.set(name); $("ex").value = name; paused = false; },
  pause() { paused = true; },
});
$("ex").onchange = () => agent.exec({ tool: "start_exercise", args: { name: $("ex").value } });
$("txt").onclick = () => { const t = $("prompt").value.trim(); if (t) { utter(t); hud(); $("prompt").value = ""; } };

// ---------------------------------------------------------------- pose loop: camera -> landmarks -> reps -> agent
function draw(g, lm, w, h) {
  const hot = S.flash && S.flash.until > performance.now() ? S.flash.j : new Set();
  g.lineWidth = 4;
  for (const [a, b] of SKELETON) {
    const A = lm[a], B = lm[b]; if ((A.visibility ?? 1) < 0.4 || (B.visibility ?? 1) < 0.4) continue;
    g.strokeStyle = hot.has(a) || hot.has(b) ? "#ef4444" : "#22c55e";
    g.beginPath(); g.moveTo(A.x * w, A.y * h); g.lineTo(B.x * w, B.y * h); g.stroke();
  }
  for (const i of PTS) {
    const P = lm[i]; if ((P.visibility ?? 1) < 0.4) continue;
    g.fillStyle = hot.has(i) ? "#ef4444" : "#fff"; g.beginPath(); g.arc(P.x * w, P.y * h, 5, 0, 7); g.fill();
  }
}
function loop() {
  if (!S) return;
  S.raf = requestAnimationFrame(loop);
  const v = $("cam"); if (v.readyState < 2 || v.currentTime === S.t) return; S.t = v.currentTime;
  const c = $("ov"); if (c.width !== v.videoWidth) { c.width = v.videoWidth; c.height = v.videoHeight; }
  const g = c.getContext("2d"); g.clearRect(0, 0, c.width, c.height);
  const r = detect(v, performance.now());
  if (!r) return warn("I can't see you. Step back so your whole body is in frame.");
  draw(g, r.img, c.width, c.height);
  if (paused) return;
  const res = engine.update(r.world, performance.now());
  if (!res.ok) return warn("Turn side-on so I can see your hips, knees and ankles.");
  if (S.warn) { S.warn = false; setState(state); }
  if (res.event) {
    const ev = res.event;
    if (ev.errors.length) S.flash = { j: new Set(ev.joints), until: performance.now() + 1500 };
    $("diag").textContent = JSON.stringify(ev);
    agent.onRep(ev); hud();
  }
}
function warn(t) { S.warn = true; $("hint").textContent = t; }

// ---------------------------------------------------------------- microphone
const MIC_FLAGS = { echoCancellation: true, noiseSuppression: true, autoGainControl: true };
const gum = (c) => navigator.mediaDevices.getUserMedia(c);
// Open the mic ourselves (with fallbacks) so a picky default doesn't kill voice commands.
async function openMic() {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("this browser has no microphone API (needs https:// and a current browser)");
  let err;
  for (let round = 0; round < 2; round++) {
    for (const audio of [MIC_FLAGS, true, { deviceId: { ideal: "default" } }]) {
      try { return await gum({ audio }); }
      catch (e) { err = e; if (e.name === "NotAllowedError" || e.name === "SecurityError") throw e; }
    }
    const devs = (await navigator.mediaDevices.enumerateDevices().catch(() => [])).filter((d) => d.kind === "audioinput" && d.deviceId);
    for (const d of devs) { try { return await gum({ audio: { deviceId: { exact: d.deviceId } } }); } catch (e) { err = e; } }
    if (round === 0) await new Promise((r) => setTimeout(r, 500));   // an input that is still waking up (Bluetooth, USB, a device switch) often appears a moment later
  }
  const devs = (await navigator.mediaDevices.enumerateDevices().catch(() => [])).filter((d) => d.kind === "audioinput");
  const names = devs.map((d) => d.label || "unnamed").join(", ");
  const e = new Error(`${err?.name}: ${err?.message} (audio inputs found: ${devs.length}${names ? " - " + names : ""})`); e.name = err?.name; throw e;
}
function micHint(e) {
  const n = e?.name;
  if (n === "NotAllowedError" || n === "SecurityError") return " Allow the microphone for this site in your browser's site settings, then press Retry.";
  if (n === "NotFoundError" || n === "OverconstrainedError") return " The browser can't open an input device. Check System Settings > Sound > Input has a working microphone selected, quit apps that may hold it (Zoom, Teams, FaceTime), then press Retry.";
  if (n === "NotReadableError") return " Another app may be using the microphone. Close it, then press Retry.";
  return "";
}
let voiceStarting = false;
async function startVoice(mic) {
  if (!S || vad || voiceStarting || !$("voiceOn").checked) return;
  voiceStarting = true; note("micNote", "");
  const cleanup = () => {
    try { vad?.destroy(); } catch {} vad = null;
    micStream?.getTracks().forEach((t) => t.stop()); micStream = null;
  };
  try {
    const live = micStream?.getAudioTracks().some((t) => t.readyState === "live");
    micStream = mic || (live ? micStream : await openMic());
    if (!S) return cleanup();
    vad = await window.vad.MicVAD.new({                       // Silero VAD: only wake Moonshine on real speech
      baseAssetPath: VAD_URL, onnxWASMBasePath: ORT_URL,
      positiveSpeechThreshold: 0.4, negativeSpeechThreshold: 0.25,   // easier to trigger on quiet or distant speech
      redemptionMs: 600,                                       // end your turn after 0.6 s of quiet (default is longer)
      minSpeechMs: 150, preSpeechPadMs: 300,                   // catch short commands, keep the first syllable
      onFrameProcessed: (p) => {                               // live meter so you can SEE the mic is being heard
        micLevel = p.isSpeech;
        const now = performance.now();
        if (S && !S.warn && state === "listening" && now - lastMeter > 120) {
          lastMeter = now; const n = Math.round(p.isSpeech * 10);
          $("hint").textContent = "Mic live  " + "▮".repeat(n) + "▯".repeat(10 - n);
        }
      },
      getStream: async () => micStream,                        // use our stream instead of the VAD's own getUserMedia
      pauseStream: async () => {},
      resumeStream: async () => micStream,
      onSpeechStart: () => { if (S && !busyTTS) setState("hearing"); },
      onVADMisfire: () => { if (S && !busyTTS) setState("listening"); },
      onSpeechEnd: (audio) => {                                // Float32Array @ 16 kHz
        if (!S) return;
        if (busyTTS) return;                                   // ignore our own voice
        setState("thinking"); moon.postMessage({ type: "transcribe", audio }, [audio.buffer]);
      },
    });
    if (!S) return cleanup();
    vad.start();
  } catch (e) {
    cleanup();
    note("micNote", "Voice commands unavailable: " + (e.message || e) + micHint(e));
  } finally { voiceStarting = false; }
}
$("retryMic").onclick = () => startVoice();
navigator.mediaDevices?.addEventListener?.("devicechange", () => { if (S && !vad) startVoice(); });   // a mic plugged in / switched mid-session is picked up

// ---------------------------------------------------------------- session
async function begin() {
  if (S || starting || ready < need) return;
  starting = true; refresh(); stopSpeech();
  unlockAudio();                                                // must happen inside the click so replies can always play (Safari)
  const video = { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } };
  let stream, mic = null;
  if ($("voiceOn").checked) {                                   // camera + mic in ONE request: one prompt, and the mic is listed straight away
    try {
      const both = await gum({ video, audio: MIC_FLAGS });
      stream = new MediaStream(both.getVideoTracks());
      if (both.getAudioTracks().length) mic = new MediaStream(both.getAudioTracks());
    } catch {}
  }
  if (!stream && !mic && $("voiceOn").checked) {                // combined request failed: try the mic on its own, before the camera opens
    try { mic = await gum({ audio: MIC_FLAGS }); } catch {}
  }
  if (!stream) {
    try { stream = await gum({ video, audio: false }); }
    catch (e) { starting = false; refresh(); return setStatus("Camera access failed: " + e.message, "error"); }
  }
  const v = $("cam"); v.srcObject = stream; try { await v.play(); } catch {}
  $("stage").classList.add("cam");
  S = { stream, t: -1, flash: null, warn: false, raf: 0 };
  engine.set($("ex").value); agent.ex = $("ex").value; paused = false;
  await startVoice(mic);
  starting = false;
  $("mic").classList.add("on"); $("micLabel").textContent = "End session";
  setState("listening"); hud();
  loop();
}
function end() {
  const s = S; if (!s) return; S = null;
  cancelAnimationFrame(s.raf); s.stream.getTracks().forEach((t) => t.stop());
  try { vad?.destroy(); } catch {} vad = null;
  micStream?.getTracks().forEach((t) => t.stop()); micStream = null;
  stopSpeech(); clearTimeout(agent.restT);
  note("micNote", ""); note("audioNote", "");
  $("cam").srcObject = null; $("stage").classList.remove("cam"); $("ov").getContext("2d").clearRect(0, 0, 9999, 9999);
  $("mic").classList.remove("on"); $("micLabel").textContent = "Start session";
  setState("off");
}
$("mic").onclick = () => (S ? end() : begin());
setState("off");

$("dl").onclick = () => {                           // session trace = future training / eval data
  const a = document.createElement("a");
  const meta = { meta: { model: llmModel, default_model: LLM_ID, profile: PROFILE.name, saved: new Date().toISOString() } };   // entries without "state" are ignored by gen_data.py --logs
  a.href = URL.createObjectURL(new Blob([JSON.stringify([meta, ...agent.log], null, 1)], { type: "application/json" }));
  a.download = "trainer-session.json"; a.click(); URL.revokeObjectURL(a.href);
};

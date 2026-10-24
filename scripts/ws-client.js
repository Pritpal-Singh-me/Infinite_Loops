/**
 * Infinite Loop – Live Analysis Client v3
 * ========================================
 * Features:
 *  - RawNet2 + AASIST voice scoring
 *  - Whisper + Groq LLM conversation analysis
 *  - Live Risk Meter (always visible, animates)
 *  - Critical Popup (score > 80) with 1930 + cybercrime.gov.in
 *  - PDF report download at end of call
 *  - Web Speech API for real-time transcript
 *  - Plain-English signal labels
 */

(function () {
  'use strict';

  const WS_URL        = 'ws://localhost:8000/ws/analyse';
  const SAMPLE_RATE   = 16000;
  const CHUNK_SECONDS = 3;
  const CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS;

  // State
  let ws           = null;
  let audioCtx     = null;
  let mediaStream  = null;
  let processor    = null;
  let sampleBuffer = [];
  let isRunning    = false;
  let recognition  = null;
  let lastTranscript = '';
  let sessionData  = {
    chunks: [], auraHistory: [], startTime: null, finalData: null
  };
  let criticalAlertShown = false;

  const $ = id => document.getElementById(id);

  // ══════════════════════════════════════════════════════════════════════════
  // LIVE RISK METER — inject into page
  // ══════════════════════════════════════════════════════════════════════════

  function injectLiveRiskMeter() {
    if ($('live-risk-meter')) return;

    const meter = document.createElement('div');
    meter.id = 'live-risk-meter';
    meter.innerHTML = `
      <div class="risk-meter-inner">
        <div class="risk-meter-header">
          <span class="risk-meter-dot" id="risk-dot"></span>
          <span class="risk-meter-label">LIVE RISK MONITOR</span>
          <span class="risk-meter-mode" id="risk-mode">INACTIVE</span>
        </div>
        <div class="risk-meter-bar-wrap">
          <div class="risk-meter-bar" id="risk-bar">
            <div class="risk-meter-fill" id="risk-fill"></div>
            <div class="risk-meter-needle" id="risk-needle"></div>
          </div>
          <div class="risk-meter-labels">
            <span>SAFE</span><span>CAUTION</span><span>HIGH</span><span>CRITICAL</span>
          </div>
        </div>
        <div class="risk-meter-score-row">
          <div class="risk-score-block">
            <span class="risk-score-num" id="meter-aura-score">—</span>
            <span class="risk-score-lbl">AURA Score</span>
          </div>
          <div class="risk-score-block">
            <span class="risk-score-num" id="meter-voice-score">—</span>
            <span class="risk-score-lbl">Voice Check</span>
          </div>
          <div class="risk-score-block">
            <span class="risk-score-num" id="meter-convo-score">—</span>
            <span class="risk-score-lbl">Convo Risk</span>
          </div>
          <div class="risk-score-block">
            <span class="risk-score-num risk-tier-badge" id="meter-tier">—</span>
            <span class="risk-score-lbl">Risk Level</span>
          </div>
        </div>
        <div class="risk-meter-verdict" id="meter-verdict">Start analysis to monitor call</div>
        <div class="risk-meter-transcript" id="meter-transcript" style="display:none"></div>
        <div class="risk-meter-flags" id="meter-flags" style="display:none"></div>
        <div class="risk-meter-actions">
          <button class="btn-risk-start" id="btn-start-analysis">▶ Start Live Analysis</button>
          <button class="btn-risk-pdf" id="btn-download-pdf" style="display:none">📄 Download Report</button>
        </div>
        <div class="risk-status-row">
          <span class="risk-dot-small" id="live-dot"></span>
          <span id="ws-status-text" class="ws-status-badge ws-status-info">Backend offline — run: cd backend && bash start.sh</span>
        </div>
      </div>
    `;

    // Insert right after hero section
    const hero = document.getElementById('hero');
    if (hero && hero.parentNode) {
      hero.parentNode.insertBefore(meter, hero.nextSibling);
    } else {
      document.body.insertBefore(meter, document.body.firstChild);
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  // INJECT STYLES
  // ══════════════════════════════════════════════════════════════════════════

  function injectStyles() {
    if ($('il-dynamic-styles')) return;
    const style = document.createElement('style');
    style.id = 'il-dynamic-styles';
    style.textContent = `
      /* ── Live Risk Meter ── */
      #live-risk-meter {
        position: relative; z-index: 20;
        margin: 0 auto 0; max-width: 880px;
        padding: 0 20px 32px;
      }
      .risk-meter-inner {
        background: rgba(10,12,18,0.92);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 20px 24px;
        backdrop-filter: blur(20px);
      }
      .risk-meter-header {
        display: flex; align-items: center; gap: 10px;
        margin-bottom: 16px;
      }
      .risk-meter-dot {
        width:10px; height:10px; border-radius:50%;
        background:#333; flex-shrink:0;
        transition: background 0.4s;
      }
      .risk-meter-dot.active {
        background:#ef4444;
        box-shadow: 0 0 12px rgba(239,68,68,0.7);
        animation: pulse-red 1s ease-in-out infinite;
      }
      @keyframes pulse-red {
        0%,100%{opacity:1;transform:scale(1)}
        50%{opacity:0.5;transform:scale(0.7)}
      }
      .risk-meter-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.70rem; letter-spacing: 0.15em;
        color: rgba(255,255,255,0.5);
        flex: 1;
      }
      .risk-meter-mode {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem; letter-spacing: 0.10em;
        color: rgba(255,255,255,0.25);
        padding: 3px 8px;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
      }

      /* Bar */
      .risk-meter-bar-wrap { margin-bottom: 6px; }
      .risk-meter-bar {
        position: relative; height: 12px;
        background: linear-gradient(to right, #10b981 0%, #facc15 40%, #f97316 70%, #ef4444 100%);
        border-radius: 6px; overflow: visible;
      }
      .risk-meter-fill {
        position: absolute; top:0; left:0; height:100%;
        background: rgba(0,0,0,0.55); border-radius: 6px;
        transition: width 0.6s cubic-bezier(.4,0,.2,1);
        width: 100%; /* overlay from right, hides the coloured part */
        transform-origin: right;
      }
      .risk-meter-needle {
        position: absolute; top: -6px;
        width: 3px; height: 24px;
        background: #fff;
        border-radius: 2px;
        box-shadow: 0 0 8px rgba(255,255,255,0.8);
        transition: left 0.6s cubic-bezier(.4,0,.2,1);
        left: 0;
      }
      .risk-meter-labels {
        display: flex; justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.60rem; letter-spacing: 0.08em;
        color: rgba(255,255,255,0.28);
        margin-top: 6px;
      }

      /* Score blocks */
      .risk-meter-score-row {
        display: grid; grid-template-columns: repeat(4, 1fr);
        gap: 10px; margin: 16px 0 12px;
      }
      .risk-score-block {
        display: flex; flex-direction: column; align-items: center;
        padding: 10px 6px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 10px;
      }
      .risk-score-num {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem; font-weight: 700;
        color: #fff; line-height: 1;
        transition: color 0.4s;
      }
      .risk-score-lbl {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.58rem; letter-spacing: 0.10em;
        color: rgba(255,255,255,0.30);
        margin-top: 4px; text-align: center;
      }
      .risk-tier-badge {
        font-size: 1rem !important; padding: 4px 10px;
        border-radius: 6px; letter-spacing: 0.08em;
      }

      /* Verdict */
      .risk-meter-verdict {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.88rem; color: rgba(255,255,255,0.55);
        text-align: center; margin-bottom: 10px;
        min-height: 22px;
        transition: color 0.4s;
      }

      /* Transcript */
      .risk-meter-transcript {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.80rem; color: rgba(255,255,255,0.50);
        font-style: italic;
        background: rgba(255,255,255,0.03);
        border-left: 2px solid rgba(255,255,255,0.12);
        padding: 8px 12px; border-radius: 4px;
        margin-bottom: 8px;
        max-height: 80px; overflow-y: auto;
      }

      /* Flags */
      .risk-meter-flags {
        display: flex; flex-wrap: wrap; gap: 6px;
        margin-bottom: 12px;
      }
      .flag-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem; padding: 3px 10px;
        background: rgba(239,68,68,0.15);
        border: 1px solid rgba(239,68,68,0.35);
        border-radius: 20px; color: #fca5a5;
        letter-spacing: 0.04em;
      }

      /* Action buttons */
      .risk-meter-actions {
        display: flex; gap: 10px; margin-bottom: 12px;
      }
      .btn-risk-start {
        flex: 1; padding: 11px 20px;
        background: #fff; color: #07080b;
        border: none; border-radius: 8px;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.82rem; font-weight: 700;
        cursor: pointer; transition: all 0.2s;
      }
      .btn-risk-start:hover { background: #e5e7eb; transform: translateY(-1px); }
      .btn-risk-start.stop { background: #ef4444; color: #fff; }
      .btn-risk-start.stop:hover { background: #dc2626; }
      .btn-risk-pdf {
        padding: 11px 18px;
        background: rgba(255,255,255,0.06);
        color: rgba(255,255,255,0.8);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 8px;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.82rem; font-weight: 600;
        cursor: pointer; transition: all 0.2s;
      }
      .btn-risk-pdf:hover { background: rgba(255,255,255,0.12); }

      /* Status row */
      .risk-status-row {
        display: flex; align-items: center; gap: 8px;
      }
      .risk-dot-small {
        width: 7px; height: 7px; border-radius: 50%;
        background: #444; flex-shrink: 0;
        transition: background 0.3s;
      }
      .risk-dot-small.active {
        background: #10b981;
        box-shadow: 0 0 8px rgba(16,185,129,0.6);
      }

      /* ── Critical Alert Popup ── */
      #critical-alert-overlay {
        position: fixed; inset: 0; z-index: 9999;
        background: rgba(0,0,0,0.82);
        display: flex; align-items: center; justify-content: center;
        animation: fade-in 0.3s ease;
      }
      @keyframes fade-in { from{opacity:0} to{opacity:1} }
      #critical-alert-box {
        background: #0a0c12;
        border: 2px solid #ef4444;
        border-radius: 16px;
        max-width: 500px; width: 90%;
        padding: 32px 28px;
        box-shadow: 0 0 60px rgba(239,68,68,0.4);
        animation: shake 0.5s ease;
        position: relative;
      }
      @keyframes shake {
        0%,100%{transform:translateX(0)}
        20%{transform:translateX(-8px)}
        40%{transform:translateX(8px)}
        60%{transform:translateX(-6px)}
        80%{transform:translateX(6px)}
      }
      .alert-icon { font-size: 3rem; text-align: center; margin-bottom: 12px; }
      .alert-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem; font-weight: 800;
        color: #ef4444; text-align: center; margin-bottom: 8px;
        letter-spacing: 0.05em;
      }
      .alert-score {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.5rem; font-weight: 700;
        color: #fff; text-align: center; margin-bottom: 12px;
      }
      .alert-message {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.90rem; color: rgba(255,255,255,0.75);
        text-align: center; line-height: 1.6; margin-bottom: 20px;
      }
      .alert-helpline {
        background: rgba(239,68,68,0.12);
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 10px; padding: 14px 18px;
        margin-bottom: 20px; text-align: center;
      }
      .alert-helpline-num {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem; font-weight: 800;
        color: #fff; letter-spacing: 0.1em;
        display: block; margin-bottom: 4px;
      }
      .alert-helpline-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem; color: rgba(255,255,255,0.4);
        letter-spacing: 0.08em;
      }
      .alert-helpline-link {
        display: block; margin-top: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem; color: #60a5fa;
        text-decoration: underline;
      }
      .alert-buttons { display: flex; gap: 10px; }
      .btn-alert-dismiss {
        flex: 1; padding: 12px;
        background: #ef4444; color: #fff;
        border: none; border-radius: 8px;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.88rem; font-weight: 700;
        cursor: pointer;
      }
      .btn-alert-report {
        flex: 1; padding: 12px;
        background: rgba(255,255,255,0.08);
        color: #fff; border: 1px solid rgba(255,255,255,0.2);
        border-radius: 8px;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.88rem; font-weight: 600;
        cursor: pointer; text-decoration: none;
        text-align: center; line-height: 1;
        display: flex; align-items: center; justify-content: center;
      }

      /* WS status badge */
      .ws-status-badge { font-family: 'JetBrains Mono', monospace; font-size: 0.70rem; letter-spacing: 0.04em; }
      .ws-status-info  { color: rgba(255,255,255,0.35); }
      .ws-status-ok    { color: #10b981; }
      .ws-status-warn  { color: #facc15; }
      .ws-status-error { color: #ef4444; }

      @media(max-width:600px) {
        .risk-meter-score-row { grid-template-columns: repeat(2,1fr); }
        .risk-meter-actions { flex-direction: column; }
      }
    `;
    document.head.appendChild(style);
  }

  // ══════════════════════════════════════════════════════════════════════════
  // UPDATE RISK METER
  // ══════════════════════════════════════════════════════════════════════════

  function updateRiskMeter(aura, voice, convo) {
    const score    = aura.aura_score || 0;
    const tier     = aura.tier || 'LOW';
    const verdict  = aura.verdict || '';
    const color    = aura.color || '#10b981';

    // Needle & fill
    const pct = score / 100;
    const fill = $('risk-fill');
    const needle = $('risk-needle');
    const bar = $('risk-bar');
    if (fill && bar) {
      // "fill" is a dark overlay from RIGHT side — shows colored bar up to score
      fill.style.left  = (pct * 100) + '%';
      fill.style.width = ((1 - pct) * 100) + '%';
    }
    if (needle && bar) {
      needle.style.left = `calc(${pct * 100}% - 1.5px)`;
    }

    // Score displays
    const setEl = (id, val, clr) => {
      const el = $(id);
      if (el) { el.textContent = val; if (clr) el.style.color = clr; }
    };
    setEl('meter-aura-score', score, color);
    setEl('meter-voice-score', Math.round(voice.score || 0));
    setEl('meter-convo-score', Math.round(convo.score || 0));

    const tierEl = $('meter-tier');
    if (tierEl) {
      tierEl.textContent = tier;
      tierEl.style.color = color;
      tierEl.style.background = color + '18';
    }

    const verdictEl = $('meter-verdict');
    if (verdictEl) {
      verdictEl.textContent = verdict;
      verdictEl.style.color = color;
    }

    // Transcript
    const transcriptEl = $('meter-transcript');
    const t = convo.transcript || '';
    if (t && transcriptEl) {
      transcriptEl.style.display = 'block';
      transcriptEl.textContent = '"' + t.slice(-200) + '"';
    }

    // LLM flags
    const flagsEl = $('meter-flags');
    const flags = convo.flags || [];
    if (flagsEl) {
      if (flags.length > 0) {
        flagsEl.style.display = 'flex';
        flagsEl.innerHTML = '⚠ ' + flags.map(f =>
          `<span class="flag-pill">${f}</span>`
        ).join('');
      } else {
        flagsEl.style.display = 'none';
      }
    }

    // Dot
    const dot = $('risk-dot');
    if (dot) dot.className = 'risk-meter-dot' + (isRunning ? ' active' : '');

    // Mode badge
    const modeEl = $('risk-mode');
    if (modeEl) modeEl.textContent = isRunning ? 'ANALYSING' : 'INACTIVE';

    // Sync main AURA section sliders
    syncMainAura(aura, voice, convo);

    // Critical alert
    if (score > 80 && !criticalAlertShown) {
      showCriticalAlert(score, verdict);
    }
  }

  function syncMainAura(aura, voice, convo) {
    const set = (id, val) => { const el = $(id); if (el) el.textContent = val; };
    const setV = (id, val) => { const el = $(id); if (el) el.value = val; };

    set('aura-score-display', `${aura.aura_score || 0} / 100`);
    set('aura-verdict-text',  `Verdict: ${aura.verdict || ''}`);
    set('voice-val-readout', Math.round(voice.score || 0) + '%');
    set('convo-val-readout', Math.round(convo.score || 0) + '%');
    setV('voice-score-input', Math.round(voice.score || 0));
    setV('convo-score-input', Math.round(convo.score || 0));

    // Highlight active risk card
    const tierMap = { LOW: 'low', MEDIUM: 'medium', HIGH: 'high', CRITICAL: 'critical' };
    document.querySelectorAll('.risk-card').forEach(c => c.classList.remove('active-tier'));
    const ac = document.querySelector(`.risk-card-${tierMap[aura.tier] || 'low'}`);
    if (ac) ac.classList.add('active-tier');
  }

  // ══════════════════════════════════════════════════════════════════════════
  // CRITICAL ALERT POPUP
  // ══════════════════════════════════════════════════════════════════════════

  function showCriticalAlert(score, verdict) {
    criticalAlertShown = true;
    if ($('critical-alert-overlay')) return;

    const overlay = document.createElement('div');
    overlay.id = 'critical-alert-overlay';
    overlay.innerHTML = `
      <div id="critical-alert-box">
        <div class="alert-icon">🚨</div>
        <div class="alert-title">VOICE CLONING DETECTED</div>
        <div class="alert-score">${score} / 100</div>
        <div class="alert-message">
          ${verdict}<br><br>
          <strong>DO NOT share OTP, bank details, Aadhaar, or transfer money.</strong><br>
          Hang up immediately and report this call.
        </div>
        <div class="alert-helpline">
          <span class="alert-helpline-num">📞 1930</span>
          <span class="alert-helpline-sub">NATIONAL CYBER CRIME HELPLINE</span>
          <a class="alert-helpline-link" href="https://cybercrime.gov.in" target="_blank" rel="noopener">
            cybercrime.gov.in →
          </a>
        </div>
        <div class="alert-buttons">
          <button class="btn-alert-dismiss" id="btn-alert-dismiss">I Understand — Close</button>
          <a class="btn-alert-report" href="https://cybercrime.gov.in" target="_blank" rel="noopener">
            Report Online 🔗
          </a>
        </div>
      </div>
    `;

    document.body.appendChild(overlay);
    $('btn-alert-dismiss').addEventListener('click', () => overlay.remove());

    // Auto-dismiss after 60 seconds
    setTimeout(() => { if ($('critical-alert-overlay')) overlay.remove(); }, 60000);
  }

  // ══════════════════════════════════════════════════════════════════════════
  // PDF DOWNLOAD
  // ══════════════════════════════════════════════════════════════════════════

  function requestPDF() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setStatus('Connect first to generate report', 'warn');
      return;
    }
    ws.send(JSON.stringify({ type: 'generate_pdf' }));
    setStatus('Generating PDF report…', 'info');
    const btn = $('btn-download-pdf');
    if (btn) { btn.textContent = '⏳ Generating…'; btn.disabled = true; }
  }

  function downloadPDF(base64Data) {
    const binary = atob(base64Data);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const blob = new Blob([bytes], { type: 'application/pdf' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url;
    a.download = `InfiniteLoop_Report_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.pdf`;
    a.click();
    URL.revokeObjectURL(url);

    const btn = $('btn-download-pdf');
    if (btn) { btn.textContent = '✓ Downloaded'; btn.disabled = false; }
  }

  // ══════════════════════════════════════════════════════════════════════════
  // STATUS
  // ══════════════════════════════════════════════════════════════════════════

  function setStatus(msg, type = 'info') {
    const badges = document.querySelectorAll('.ws-status-badge, [id="ws-status-text"]');
    badges.forEach(el => {
      el.textContent = msg;
      el.className = 'ws-status-badge ws-status-' + type;
    });
    const dots = document.querySelectorAll('.live-dot, .risk-dot-small, [id="live-dot"]');
    dots.forEach(dot => {
      const isSmall = dot.classList.contains('risk-dot-small');
      dot.className = (isSmall ? 'risk-dot-small' : 'live-dot') + (type === 'ok' ? ' active' : '');
    });
  }

  function updateButtons(running) {
    const btns = document.querySelectorAll('.btn-live-analysis, .btn-risk-start, [id="btn-start-analysis"]');
    btns.forEach(btn => {
      if (running) {
        btn.innerHTML = '⏹ Stop Analysis';
        btn.classList.add('stop');
      } else {
        btn.innerHTML = '▶ Start Live Analysis';
        btn.classList.remove('stop');
      }
    });
    const dots = document.querySelectorAll('#risk-dot, .risk-meter-dot');
    dots.forEach(d => {
      d.className = running ? 'risk-meter-dot active' : 'risk-meter-dot';
    });
    const modeEl = $('risk-mode');
    if (modeEl && running) modeEl.textContent = 'ANALYSING';
  }

  // ══════════════════════════════════════════════════════════════════════════
  // WEB SPEECH API
  // ══════════════════════════════════════════════════════════════════════════

  function startSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return;

    recognition = new SpeechRecognition();
    recognition.continuous     = true;
    recognition.interimResults = true;
    recognition.lang           = 'en-IN';

    recognition.onresult = (event) => {
      let final = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (event.results[i].isFinal) final += event.results[i][0].transcript + ' ';
      }
      if (final && ws && ws.readyState === WebSocket.OPEN) {
        lastTranscript = (lastTranscript + ' ' + final).trim().slice(-800);
        ws.send(JSON.stringify({ type: 'transcript', text: lastTranscript }));

        // Show in meter transcript
        const tel = $('meter-transcript');
        if (tel) { tel.style.display = 'block'; tel.textContent = '"' + lastTranscript.slice(-200) + '"'; }
      }
    };

    recognition.onend = () => {
      if (isRunning) setTimeout(() => { try { recognition.start(); } catch(e){} }, 200);
    };

    try { recognition.start(); } catch(e) {}
  }

  function stopSpeechRecognition() {
    if (recognition) { try { recognition.stop(); } catch(e){} recognition = null; }
    lastTranscript = '';
  }

  // ══════════════════════════════════════════════════════════════════════════
  // WEBSOCKET
  // ══════════════════════════════════════════════════════════════════════════

  function connectWS(onOpen) {
    setStatus('Connecting…', 'info');
    ws = new WebSocket(WS_URL);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      setStatus('Connected ✓ — listening every 3s', 'ok');
      ws.send('ping');
      if (onOpen) onOpen();
    };

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === 'analysis_result') {
          // Update risk meter
          updateRiskMeter(msg.aura, msg.voice, msg.conversation);
          // Show PDF button after first chunk
          const pdfBtn = $('btn-download-pdf');
          if (pdfBtn) pdfBtn.style.display = 'inline-flex';
        } else if (msg.type === 'pdf_ready') {
          downloadPDF(msg.pdf_b64);
          setStatus('PDF downloaded ✓', 'ok');
        } else if (msg.type === 'pdf_error') {
          setStatus('PDF error: ' + msg.message, 'error');
          const btn = $('btn-download-pdf');
          if (btn) { btn.textContent = '📄 Download Report'; btn.disabled = false; }
        } else if (msg.type === 'error') {
          setStatus('Backend error: ' + msg.message, 'error');
        }
      } catch(e) {}
    };

    ws.onerror = () => setStatus('Backend unreachable — run: cd backend && bash start.sh', 'error');
    ws.onclose = () => {
      if (isRunning) {
        setStatus('Reconnecting…', 'warn');
        setTimeout(() => {
          if (isRunning) {
            connectWS(async () => {
              if (!mediaStream) await startMic();
              startSpeechRecognition();
            });
          }
        }, 3000);
      }
    };
  }

  function sendAudioChunk(samples) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const int16 = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      int16[i] = Math.max(-32768, Math.min(32767, Math.round(samples[i] * 32767)));
    }
    ws.send(int16.buffer);
  }

  // ══════════════════════════════════════════════════════════════════════════
  // MICROPHONE
  // ══════════════════════════════════════════════════════════════════════════

  async function startMic() {
    if (mediaStream && audioCtx && processor) return;
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Microphone access requires HTTPS or http://localhost');
      }
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, sampleRate: SAMPLE_RATE,
                 echoCancellation: false, noiseSuppression: false, autoGainControl: false }
      });
      audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
      const source = audioCtx.createMediaStreamSource(mediaStream);
      processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = (e) => {
        if (!isRunning) return;
        const s = e.inputBuffer.getChannelData(0);
        for (let i = 0; i < s.length; i++) sampleBuffer.push(s[i]);
        if (sampleBuffer.length >= CHUNK_SAMPLES) {
          sendAudioChunk(sampleBuffer.splice(0, CHUNK_SAMPLES));
        }
      };
      source.connect(processor);
      processor.connect(audioCtx.destination);
    } catch(err) {
      setStatus('Mic notice: ' + err.message, 'warn');
    }
  }

  function stopMic() {
    if (processor)   { processor.disconnect(); processor = null; }
    if (audioCtx)    { audioCtx.close(); audioCtx = null; }
    if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
    sampleBuffer = [];
  }

  // ══════════════════════════════════════════════════════════════════════════
  // START / STOP
  // ══════════════════════════════════════════════════════════════════════════

  async function startAnalysis() {
    if (isRunning) return;
    isRunning = true;
    criticalAlertShown = false;
    sessionData = { chunks: [], auraHistory: [], startTime: Date.now(), finalData: null };

    updateButtons(true);

    connectWS(async () => {
      await startMic();
      startSpeechRecognition();
    });
  }

  function stopAnalysis() {
    isRunning = false;
    stopMic();
    stopSpeechRecognition();
    if (ws) { try { ws.close(); } catch(e){} ws = null; }

    updateButtons(false);
    setStatus('Analysis stopped. Use "Download Report" for PDF.', 'info');
    const modeEl = $('risk-mode');
    if (modeEl) modeEl.textContent = 'INACTIVE';
  }

  // ══════════════════════════════════════════════════════════════════════════
  // HEALTH CHECK
  // ══════════════════════════════════════════════════════════════════════════

  function pollHealth() {
    fetch('http://localhost:8000/health', { signal: AbortSignal.timeout(2500) })
      .then(r => r.json())
      .then(d => {
        const parts = [];
        if (d.models)  parts.push('RawNet2+AASIST');
        if (d.whisper) parts.push('Whisper');
        if (d.groq)    parts.push('Groq LLM');
        if (!isRunning) {
          setStatus(
            `Backend ready ✓ — ${parts.length ? parts.join(', ') : 'heuristic mode'}`,
            'ok'
          );
          const modeEl = $('risk-mode');
          if (modeEl) modeEl.textContent = d.mode ? d.mode.toUpperCase() : 'READY';
        }
      })
      .catch(() => {
        if (!isRunning) {
          setStatus('Backend offline — cd backend && bash start.sh', 'warn');
          const modeEl = $('risk-mode');
          if (modeEl) modeEl.textContent = 'OFFLINE';
        }
      });
  }

  // ══════════════════════════════════════════════════════════════════════════
  // INIT
  // ══════════════════════════════════════════════════════════════════════════

  function init() {
    injectStyles();
    injectLiveRiskMeter();

    setTimeout(() => {
      // Wire all start/stop buttons
      const startBtns = document.querySelectorAll('.btn-live-analysis, .btn-risk-start, [id="btn-start-analysis"]');
      startBtns.forEach(btn => {
        btn.addEventListener('click', () => {
          if (isRunning) stopAnalysis();
          else           startAnalysis();
        });
      });

      const pdfBtns = document.querySelectorAll('.btn-risk-pdf, [id="btn-download-pdf"]');
      pdfBtns.forEach(btn => btn.addEventListener('click', requestPDF));

      // Initial check & continuous poll every 3.5s
      pollHealth();
      setInterval(pollHealth, 3500);
    }, 100);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  window.InfiniteLoopClient = { startAnalysis, stopAnalysis, requestPDF, showCriticalAlert };
})();

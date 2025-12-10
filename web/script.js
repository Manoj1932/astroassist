// =====================================
//  AstroAssist Command Center Frontend
// =====================================
let oxygenLevel = 90;   // starting oxygen
let batteryLevel = 80;
let pressureLevel=100; // in kPa
let tempLevel=22;      // in °C
let fireLevel = 0;        // 0–100
let radiationLevel = 5;  // 0–100
let crewHealth = 100;    // 0–100
let autoDoorLocked = false;
let oxygenBoostActive = false;
let evacuationMode = false;
let systemShutdown = false;

const API_URL = "/predict";

const ws = new WebSocket(`ws://${location.host}/ws/sensors`);

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  oxygenLevel = data.oxygen;
  pressureLevel = data.pressure;
  powerLevel = data.power;

  // ✅ Update values on UI
  document.getElementById("oxygenValue").textContent = `${oxygenLevel}%`;
  document.getElementById("pressureValue").textContent = `${pressureLevel} kPa`;
  document.getElementById("powerValue").textContent = `${powerLevel}%`;

  // ✅ Update bars
  document.getElementById("oxygenBar").style.width = `${oxygenLevel}%`;
  document.getElementById("pressureBar").style.width = `${(pressureLevel / 110) * 100}%`;
  document.getElementById("powerBar").style.width = `${powerLevel}%`;
};

const INTENT_LABELS = [
  "check_status",
  "control_door",
  "system_control",
  "ask_info",
  "emergency",
];

// ---- DOM ELEMENTS ----
let alarmSound;
let commandForm;
let commandInput;
let sendBtn;
let lastCommandEl;
let lastIntentTextEl;
let emergencyReasonTextEl;
let historyList;
let emergencyBanner;
let emergencyBannerReason;

let oxygenValEl, pressureValEl, powerValEl, tempValEl, radiationValEl, co2ValEl;
let oxygenBar, pressureBar, powerBar, tempBar, radiationBar, co2Bar;
let oxygenChip, batteryChip, systemChip;

let radarCanvas, radarCtx;
let radarAngle = 0;

let intentChart;

// ---- STATE ----
let sensors = {
  oxygen: 98,
  pressure: 101.3,
  power: 90,
  temp: 23,
  radiation: 0.3,
  co2: 0.05,
};

let intentCounts = {
  check_status: 0,
  control_door: 0,
  system_control: 0,
  ask_info: 0,
  emergency: 0,
};

let currentEmergencyReason = null;

// ============== INIT ==============
window.addEventListener("DOMContentLoaded", () => {
  cacheDom();
  initCharts();
  initRadar();
  renderSensors();
  setInterval(updateSensorsTick, 2000);

  commandForm.addEventListener("submit", handleSubmit);
});

// Cache DOM elements once
function cacheDom() {
  alarmSound = document.getElementById("alarmSound");
  commandForm = document.getElementById("commandForm");
  commandInput = document.getElementById("commandInput");
  sendBtn = document.getElementById("sendBtn");
  lastCommandEl = document.getElementById("lastCommand");
  lastIntentTextEl = document.getElementById("lastIntentText");
  emergencyReasonTextEl = document.getElementById("emergencyReasonText");
  historyList = document.getElementById("historyList");
  emergencyBanner = document.getElementById("emergencyBanner");
  emergencyBannerReason = document.getElementById("emergencyBannerReason");

  oxygenValEl = document.getElementById("oxygenValue");
  pressureValEl = document.getElementById("pressureValue");
  powerValEl = document.getElementById("powerValue");
  tempValEl = document.getElementById("tempValue");
  radiationValEl = document.getElementById("radiationValue");
  co2ValEl = document.getElementById("co2Value");

  oxygenBar = document.getElementById("oxygenBar");
  pressureBar = document.getElementById("pressureBar");
  powerBar = document.getElementById("powerBar");
  tempBar = document.getElementById("tempBar");
  radiationBar = document.getElementById("radiationBar");
  co2Bar = document.getElementById("co2Bar");

  oxygenChip = document.getElementById("oxygenChip");
  batteryChip = document.getElementById("batteryChip");
  systemChip = document.getElementById("systemChip");

  radarCanvas = document.getElementById("radarCanvas");
  radarCtx = radarCanvas.getContext("2d");
}

// ============== CHARTS ==============
function initCharts() {
  const ctx = document.getElementById("intentChart").getContext("2d");

  intentChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: INTENT_LABELS.map((x) => x.replace("_", " ")),
      datasets: [
        {
          label: "Count",
          data: INTENT_LABELS.map(() => 0),
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          labels: {
            color: "#e5e7eb",
            font: { size: 10 },
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#e5e7eb", font: { size: 10 } },
          grid: { color: "rgba(31, 41, 55, 0.5)" },
        },
        y: {
          ticks: { color: "#9ca3af", font: { size: 9 } },
          grid: { color: "rgba(31, 41, 55, 0.3)" },
          beginAtZero: true,
          precision: 0,
        },
      },
    },
  });
}

function updateIntentChart() {
  const dataset = intentChart.data.datasets[0];
  dataset.data = INTENT_LABELS.map((label) => intentCounts[label]);
  intentChart.update();
}

// ============== RADAR ==============
function initRadar() {
  function drawRadar() {
    const w = radarCanvas.width;
    const h = radarCanvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const r = w / 2 - 10;

    radarCtx.clearRect(0, 0, w, h);

    // Background
    const radial = radarCtx.createRadialGradient(cx, cy, 0, cx, cy, r);
    radial.addColorStop(0, "rgba(15,23,42,1)");
    radial.addColorStop(1, "rgba(3,7,18,1)");
    radarCtx.fillStyle = radial;
    radarCtx.fillRect(0, 0, w, h);

    radarCtx.save();
    radarCtx.translate(cx, cy);

    // Outer circle
    radarCtx.beginPath();
    radarCtx.arc(0, 0, r, 0, Math.PI * 2);
    radarCtx.strokeStyle = "#00ffe7";
    radarCtx.lineWidth = 3;
    radarCtx.stroke();

    // Crosshair
    radarCtx.beginPath();
    radarCtx.moveTo(-r, 0);
    radarCtx.lineTo(r, 0);
    radarCtx.moveTo(0, -r);
    radarCtx.lineTo(0, r);
    radarCtx.strokeStyle = "rgba(148, 163, 184, 0.25)";
    radarCtx.lineWidth = 1;
    radarCtx.stroke();

    // Sweep
    radarAngle += 0.03;
    const sx = Math.cos(radarAngle) * (r - 8);
    const sy = Math.sin(radarAngle) * (r - 8);

    radarCtx.beginPath();
    radarCtx.moveTo(0, 0);
    radarCtx.lineTo(sx, sy);
    radarCtx.strokeStyle = "#22d3ee";
    radarCtx.lineWidth = 3;
    radarCtx.stroke();

    radarCtx.restore();
    requestAnimationFrame(drawRadar);
  }

  requestAnimationFrame(drawRadar);
}

// ============== SENSORS ==============
function updateSensorsTick() {
  // Slight drift + noise
  sensors.oxygen = clamp(sensors.oxygen + randomDelta(0.5) - 0.02, 15, 100);
  sensors.pressure = clamp(sensors.pressure + randomDelta(0.8), 85, 105);
  sensors.power = clamp(sensors.power + randomDelta(1.2) - 0.15, 10, 100);
  sensors.temp = clamp(sensors.temp + randomDelta(0.4), 18, 35);
  sensors.radiation = clamp(sensors.radiation + randomDelta(0.05), 0.1, 8);
  sensors.co2 = clamp(sensors.co2 + randomDelta(0.012), 0.03, 2.0);

  // Occasionally simulate leak / failure for demo
  if (Math.random() < 0.005) {
    sensors.oxygen = Math.max(10, sensors.oxygen - 8);
  }

  renderSensors();
  checkSensorEmergency();
}

function renderSensors() {
  oxygenValEl.textContent = `${sensors.oxygen.toFixed(1)}%`;
  pressureValEl.textContent = `${sensors.pressure.toFixed(1)} kPa`;
  powerValEl.textContent = `${Math.round(sensors.power)}%`;
  tempValEl.textContent = `${sensors.temp.toFixed(1)} °C`;
  radiationValEl.textContent = `${sensors.radiation.toFixed(2)} mSv`;
  co2ValEl.textContent = `${sensors.co2.toFixed(2)}%`;

  oxygenBar.style.width = `${sensors.oxygen}%`;
  pressureBar.style.width = `${(sensors.pressure / 110) * 100}%`;
  powerBar.style.width = `${sensors.power}%`;
  tempBar.style.width = `${((sensors.temp + 10) / 55) * 100}%`;
  radiationBar.style.width = `${(sensors.radiation / 8) * 100}%`;
  co2Bar.style.width = `${(sensors.co2 / 2) * 100}%`;

  setDanger(oxygenBar, sensors.oxygen < 30);
  setDanger(pressureBar, sensors.pressure < 90);
  setDanger(powerBar, sensors.power < 20);
  setDanger(tempBar, sensors.temp < 0 || sensors.temp > 45);
  setDanger(radiationBar, sensors.radiation > 5);
  setDanger(co2Bar, sensors.co2 > 1.5);

  // Top chips
  if (oxygenChip) {
    oxygenChip.textContent = `Oxygen: ${sensors.oxygen.toFixed(1)}%`;
  }
  if (batteryChip) {
    batteryChip.textContent = `Battery: ${Math.round(sensors.power)}%`;
  }
}

function setDanger(barEl, isDanger) {
  if (!barEl) return;
  barEl.classList.toggle("danger", isDanger);
}

function randomDelta(maxAbs) {
  return (Math.random() * 2 - 1) * maxAbs;
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

// Return a string if sensors are in emergency, else null
function getSensorEmergencyReason() {
  if (sensors.oxygen < 30) return "Low oxygen (< 30%)";
  if (sensors.pressure < 60) return "Cabin pressure low";
  if (sensors.power < 1) return "Power critically low";
  if (sensors.temp < 0 || sensors.temp > 45) return "Temperature out of range";
  if (sensors.radiation > 5) return "Radiation spike detected";
  if (sensors.co2 > 1.5) return "CO₂ level too high";
  return null;
}

function checkSensorEmergency() {
  const sensorReason = getSensorEmergencyReason();
  if (sensorReason) {
    // Auto emergency from sensors
    setEmergencyMode(sensorReason, "sensors");
  } else if (!currentEmergencyReason || currentEmergencyReason.source === "sensors") {
    // Clear if only sensors were causing it
    setEmergencyMode(null, null);
  }
}

// ============== EMERGENCY UI ==============
function setEmergencyMode(reason, source) {
  // reason = string or null, source = "model" | "sensors" | null
  const isEmergency = !!reason;
  currentEmergencyReason = isEmergency ? { text: reason, source } : null;

  if (isEmergency) {
    document.body.classList.add("emergency-active");
    emergencyReasonTextEl.textContent = `${reason} (${source === "sensors" ? "sensors" : "intent"})`;
    emergencyBanner.classList.remove("hidden");
    emergencyBannerReason.textContent = `— ${reason}`;

    systemChip.textContent = "System: ALERT";

    if (alarmSound) {
      alarmSound.currentTime = 0;
      alarmSound.play().catch(() => {});
    }
  } else {
    document.body.classList.remove("emergency-active");
    emergencyBanner.classList.add("hidden");
    emergencyBannerReason.textContent = "";
    emergencyReasonTextEl.textContent = "—";
    systemChip.textContent = "System: ONLINE";

    if (alarmSound) {
      alarmSound.pause();
    }
  }
}

// ============== INTENT + COMMAND HANDLING ==============
async function handleSubmit(e) {
  e.preventDefault();
  const text = commandInput.value.trim();
  if (!text) return;

  sendBtn.disabled = true;
  sendBtn.textContent = "Analyzing…";

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const data = await res.json();
    let modelIntent = data.predicted_intent || "unknown";

    // Sensor-aware logic:
    // If oxygen < 30 or other hazard, FORCE emergency intent
    const sensorReason = getSensorEmergencyReason();
    let displayedIntent = modelIntent;
    let emergencyReason = null;
    let emergencySource = null;

    if (modelIntent === "emergency" || oxygenLevel < 30) {
      emergencyReason = "Model detected emergency command";
      emergencySource = "model";
    }

    if (sensorReason) {
      // Oxygen / sensor-based emergency overrides everything
      displayedIntent = "emergency";
      emergencyReason = emergencyReason
        ? `${emergencyReason} + ${sensorReason}`
        : sensorReason;
      emergencySource = "sensors";
    }

    // Update intent counts & UI using displayed intent
    if (displayedIntent in intentCounts) {
      intentCounts[displayedIntent]++;
    }
    updateIntentChart();

    lastCommandEl.textContent = text;
    lastIntentTextEl.textContent = displayedIntent.replace("_", " ");
    lastIntentTextEl.className = `intent-pill intent-${displayedIntent}`;

    // Emergency UI
    if (emergencyReason) {
      setEmergencyMode(emergencyReason, emergencySource);
    } else {
      setEmergencyMode(null, null);
    }

    // History
    addHistoryItem(text, displayedIntent);

  } catch (err) {
    console.error(err);
    lastIntentTextEl.textContent = "Error";
    lastIntentTextEl.className = "intent-pill";
    emergencyReasonTextEl.textContent = "Prediction failed";
  } finally {
    sendBtn.disabled = false;
    sendBtn.textContent = "Analyze";
    commandInput.value = "";
    commandInput.focus();
  }
}

function addHistoryItem(command, intent, reason) {
  const item = document.createElement("div");
  item.className = "history-item";

  const cmdSpan = document.createElement("span");
  cmdSpan.className = "command";
  cmdSpan.textContent = command || "— AUTO SENSOR EVENT —";

  const intentSpan = document.createElement("span");
  intentSpan.className = `badge badge-intent intent-${intent}`;
  intentSpan.textContent = intent.replace("_", " ");

  const timeSpan = document.createElement("span");
  timeSpan.className = "badge badge-time";
  const now = new Date();
  timeSpan.textContent = now.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });

  item.appendChild(cmdSpan);
  item.appendChild(intentSpan);
  item.appendChild(timeSpan);

  if (reason) {
    const reasonRow = document.createElement("div");
    reasonRow.className = "mono";
    reasonRow.style.gridColumn = "1 / -1";
    reasonRow.textContent = `⚠ ${reason}`;
    item.appendChild(reasonRow);
  }

  historyList.prepend(item);
}

function updateTelemetry() {
  oxygenLevel = Math.max(0, oxygenLevel - Math.random() * 2);
  batteryLevel = Math.max(10, batteryLevel - Math.random() * 0.5);

  fireLevel = Math.random() < 0.15 ? Math.min(100, fireLevel + 5) : Math.max(0, fireLevel - 2);
  radiationLevel = Math.min(100, radiationLevel + Math.random() * 2);
  crewHealth = Math.max(0, crewHealth - (oxygenLevel < 30 ? 5 : 0));

  document.getElementById("oxygen").innerText = oxygenLevel.toFixed(1) + "%";
  document.getElementById("battery").innerText = batteryLevel.toFixed(0) + "%";

  // ✅ AUTO EMERGENCY CONDITIONS
  if (oxygenLevel < 30 || fireLevel > 70 || radiationLevel > 70) {
    activateCriticalSystemMode();
  }

  aiDecisionEngine();
}

function activateCriticalSystemMode() {
  const resultBox = document.getElementById("resultBox");
  const alarm = document.getElementById("alarmSound");

  resultBox.className = "result-box emergency";
  resultBox.innerHTML = `
    🚨 AUTO EMERGENCY 🚨<br>
    🫁 Oxygen: ${oxygenLevel.toFixed(1)}%<br>
    🔥 Fire: ${fireLevel}%<br>
    ☢ Radiation: ${radiationLevel.toFixed(1)}%<br>
    ❤️ Crew Health: ${crewHealth}%
  `;

  document.body.classList.add("emergency-active");

  if (alarm) {
    alarm.currentTime = 0;
    alarm.play().catch(() => {});
  }
}

function aiDecisionEngine() {
  const historyBox = document.getElementById("historyBox");

  // 🔥 FIRE → AUTO LOCK DOORS
  if (fireLevel > 60 && !autoDoorLocked) {
    autoDoorLocked = true;
    logAI("🔥 Fire detected → Auto Door Lock ENABLED");
    speakAlert("Warning! Fire detected. Auto door lock enabled.");
  }

  // 🫁 LOW OXYGEN → OXYGEN BOOST
  if (oxygenLevel < 30 && !oxygenBoostActive) {
    oxygenBoostActive = true;
    oxygenLevel += 15;
    logAI("🫁 Oxygen critical → Emergency Oxygen BOOST activated");
    speakAlert("Warning! Oxygen levels critical. Emergency oxygen boost activated.");
  }

  // ☢ RADIATION → SYSTEM SHUTDOWN
  if (radiationLevel > 80 && !systemShutdown) {
    systemShutdown = true;
    logAI("☢ Radiation extreme → SYSTEM SHUTDOWN initiated");
    speakAlert("Critical alert! Radiation levels extreme. Initiating system shutdown.");
    document.body.style.background = "#000";
  }

  // ❤️ CREW HEALTH → AUTO EVACUATION
  if (crewHealth < 40 && !evacuationMode) {
    evacuationMode = true;
    logAI("🚀 Crew health critical → AUTO EVACUATION triggered");
    speakAlert("Emergency! Crew health critical. Auto evacuation triggered.");
  }
}
 
function logAI(message) {
  const historyBox = document.getElementById("historyBox");
  const div = document.createElement("div");
  div.innerHTML = "🤖 AI SYSTEM → <b>" + message + "</b>";
  div.style.color = "orange";
  historyBox.appendChild(div);
}

function speakAlert(text) {
  const msg = new SpeechSynthesisUtterance(text);
  msg.rate = 1;
  msg.pitch = 1;
  window.speechSynthesis.speak(msg);
}

// Run every 2 seconds
setInterval(updateTelemetry, 2000);
/**
 * =============================================================================
 * CardioPredict - Frontend Client Controller (with Authentication)
 * =============================================================================
 * Handles:
 * 1. Authentication State: Login, Registration, Session Persistence (localStorage), Logout
 * 2. Form input gathering & client-side validation for 13 clinical biomarkers
 * 3. Communication with the Flask ML REST API via fetch()
 * 4. Rendering dynamic prediction results, badges, and animated probability bars
 * 5. Error handling with friendly troubleshooting instructions
 * 6. Tab navigation, sample presets, and smooth scrolling to Biomarkers Guide
 * =============================================================================
 */

// Target API endpoints on Flask server
const API_BASE_URL = "http://127.0.0.1:5000";
const PREDICT_ENDPOINT = `${API_BASE_URL}/predict`;
const HEALTH_ENDPOINT = `${API_BASE_URL}/`;
const LOGIN_ENDPOINT = `${API_BASE_URL}/login`;
const REGISTER_ENDPOINT = `${API_BASE_URL}/register`;
const HISTORY_ENDPOINT = `${API_BASE_URL}/history`;

// Storage key for authenticated session
const AUTH_STORAGE_KEY = "cardio_user";

// Clinical feature keys expected by the Flask ML pipeline
const FEATURE_KEYS = [
  "age", "sex", "cp", "trestbps", "chol", "fbs", 
  "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
];

// Sample test profiles based on validated Cleveland records
const SAMPLE_PROFILES = {
  healthy: {
    age: 45,
    sex: "0",         // Female
    cp: "1",          // Typical Angina
    trestbps: 120,    // Normal BP
    chol: 180,        // Healthy cholesterol
    fbs: "0",         // Normal glucose (<= 120 mg/dl)
    restecg: "0",     // Normal ECG
    thalach: 175,     // High max heart rate
    exang: "0",       // No exercise angina
    oldpeak: 0.0,     // No ST depression
    slope: "1",       // Upsloping
    ca: "0",          // 0 major vessels
    thal: "3"         // Normal perfusion
  },
  atRisk: {
    age: 67,
    sex: "1",         // Male
    cp: "4",          // Asymptomatic (severe risk category)
    trestbps: 160,    // Stage 2 hypertension
    chol: 286,        // High cholesterol
    fbs: "0",         // <= 120 mg/dl
    restecg: "2",     // Left ventricular hypertrophy
    thalach: 108,     // Blunted max heart rate
    exang: "1",       // Exercise induced angina
    oldpeak: 1.5,     // Significant ST depression
    slope: "2",       // Flat ST slope
    ca: "3",          // 3 vessels colored by fluoroscopy
    thal: "3"         // Normal / fixed defect
  }
};

// =============================================================================
// DOM Elements
// =============================================================================

// Screens
const authScreen = document.getElementById("authScreen");
const dashboardScreen = document.getElementById("dashboardScreen");

// Auth Controls
const tabLoginBtn = document.getElementById("tabLoginBtn");
const tabRegisterBtn = document.getElementById("tabRegisterBtn");
const loginForm = document.getElementById("loginForm");
const registerForm = document.getElementById("registerForm");
const authAlert = document.getElementById("authAlert");
const linkToRegister = document.getElementById("linkToRegister");
const linkToLogin = document.getElementById("linkToLogin");
const quickDemoBtn = document.getElementById("quickDemoBtn");

// User Profile in Header
const userFullName = document.getElementById("userFullName");
const logoutBtn = document.getElementById("logoutBtn");

// Prediction Form & Presets
const heartForm = document.getElementById("heartForm");
const resetFormBtn = document.getElementById("resetFormBtn");
const loadHealthySampleBtn = document.getElementById("loadHealthySampleBtn");
const loadAtRiskSampleBtn = document.getElementById("loadAtRiskSampleBtn");
const scrollToGuideBtn = document.getElementById("scrollToGuideBtn");

// Result card state containers
const initialView = document.getElementById("initialView");
const loadingView = document.getElementById("loadingView");
const errorView = document.getElementById("errorView");
const resultView = document.getElementById("resultView");

const errorTitle = document.getElementById("errorTitle");
const errorMessageText = document.getElementById("errorMessageText");
const errorRetryBtn = document.getElementById("errorRetryBtn");

// Outcome display elements
const outcomeBox = document.getElementById("outcomeBox");
const outcomeIconCircle = document.getElementById("outcomeIconCircle");
const outcomeRiskBadge = document.getElementById("outcomeRiskBadge");
const outcomeHeading = document.getElementById("outcomeHeading");

// Probability bars & metrics
const noDiseasePercent = document.getElementById("noDiseasePercent");
const noDiseaseProgress = document.getElementById("noDiseaseProgress");
const diseasePercent = document.getElementById("diseasePercent");
const diseaseProgress = document.getElementById("diseaseProgress");

const rawClassVal = document.getElementById("rawClassVal");
const modelConfidenceVal = document.getElementById("modelConfidenceVal");
const riskCategoryVal = document.getElementById("riskCategoryVal");
const clinicalAdviceText = document.getElementById("clinicalAdviceText");

// API Status Chip
const apiStatusChip = document.getElementById("apiStatusChip");
const apiStatusLabel = document.getElementById("apiStatusLabel");


// =============================================================================
// Authentication System Logic
// =============================================================================

/**
 * Display alert banner in the Auth card
 * @param {string} message 
 * @param {'error'|'success'} type 
 */
function showAuthAlert(message, type = "error") {
  authAlert.className = `auth-alert ${type}`;
  const icon = type === "success" ? "✓" : "⚠️";
  authAlert.innerHTML = `<span>${icon}</span> <span>${message}</span>`;
  authAlert.style.display = "flex";
}

/**
 * Hide the auth alert banner
 */
function hideAuthAlert() {
  authAlert.style.display = "none";
  authAlert.innerHTML = "";
}

/**
 * Switch view between 'login' and 'register'
 * @param {'login'|'register'} mode 
 */
function switchAuthMode(mode) {
  hideAuthAlert();
  if (mode === "login") {
    loginForm.style.display = "flex";
    registerForm.style.display = "none";
    tabLoginBtn.classList.add("active");
    tabRegisterBtn.classList.remove("active");
  } else {
    loginForm.style.display = "none";
    registerForm.style.display = "flex";
    tabLoginBtn.classList.remove("active");
    tabRegisterBtn.classList.add("active");
  }
}

/**
 * Log in the user into the frontend session and reveal the dashboard
 * @param {Object} userData 
 */
function setAuthenticatedUser(userData) {
  try {
    localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(userData));
  } catch (e) {
    console.warn("Could not save to localStorage:", e);
  }

  // Display user's name
  if (userFullName) {
    userFullName.textContent = userData.fullname || userData.username || "Clinician";
  }

  // Transition from auth screen to dashboard
  authScreen.style.display = "none";
  dashboardScreen.style.display = "flex";

  // Check backend server status and load history
  checkServerHealth();
  fetchAndRenderHistory();
}

/**
 * Retrieve current authenticated user object from localStorage
 */
function getAuthenticatedUser() {
  try {
    const raw = localStorage.getItem(AUTH_STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (e) {
    return null;
  }
}

/**
 * Log out the user and return to the login screen
 */
function logoutUser() {
  try {
    localStorage.removeItem(AUTH_STORAGE_KEY);
  } catch (e) {
    console.warn("Could not clear localStorage:", e);
  }

  // Reset forms
  loginForm.reset();
  registerForm.reset();
  heartForm.reset();
  setResultState("initial");

  // Switch screens
  dashboardScreen.style.display = "none";
  authScreen.style.display = "flex";
  switchAuthMode("login");
  showAuthAlert("You have been signed out successfully.", "success");
}

/**
 * Check if a user session is already stored in localStorage
 */
function checkExistingSession() {
  try {
    const rawUser = localStorage.getItem(AUTH_STORAGE_KEY);
    if (rawUser) {
      const user = JSON.parse(rawUser);
      if (user && (user.username || user.email)) {
        setAuthenticatedUser(user);
        return true;
      }
    }
  } catch (e) {
    localStorage.removeItem(AUTH_STORAGE_KEY);
  }

  // Not logged in: show login screen
  dashboardScreen.style.display = "none";
  authScreen.style.display = "flex";
  switchAuthMode("login");
  return false;
}

/**
 * Handle Login Form Submission
 */
async function handleLoginSubmit(event) {
  event.preventDefault();
  hideAuthAlert();

  const identifier = document.getElementById("loginIdentifier").value.trim();
  const password = document.getElementById("loginPassword").value;

  if (!identifier || !password) {
    showAuthAlert("Please enter your username/email and password.");
    return;
  }

  const submitBtn = document.getElementById("loginSubmitBtn");
  const originalBtnText = submitBtn.innerHTML;
  submitBtn.disabled = true;
  submitBtn.innerHTML = "<span>Signing In...</span>";

  try {
    const response = await fetch(LOGIN_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username_or_email: identifier,
        password: password
      })
    });

    const data = await response.json().catch(() => null);

    if (!response.ok) {
      const msg = (data && data.error) ? data.error : `Login failed (Status: ${response.status})`;
      throw new Error(msg);
    }

    // Success: log user into dashboard
    console.log("[Auth] Logged in successfully:", data.user);
    setAuthenticatedUser(data.user);

  } catch (err) {
    console.error("[Auth] Login error:", err);
    let errorMsg = err.message || "Failed to log in.";
    if (err.name === "TypeError" && err.message.includes("fetch")) {
      errorMsg = "Unable to connect to Flask server (http://127.0.0.1:5000). Please make sure backend is running: python app.py";
    }
    showAuthAlert(errorMsg, "error");
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = originalBtnText;
  }
}

/**
 * Handle Register Form Submission
 */
async function handleRegisterSubmit(event) {
  event.preventDefault();
  hideAuthAlert();

  const fullname = document.getElementById("regFullname").value.trim();
  const username = document.getElementById("regUsername").value.trim();
  const email = document.getElementById("regEmail").value.trim();
  const password = document.getElementById("regPassword").value;
  const confirmPassword = document.getElementById("regConfirmPassword").value;

  if (!fullname || !username || !email || !password) {
    showAuthAlert("Please fill in all registration fields.");
    return;
  }

  if (password.length < 6) {
    showAuthAlert("Password must be at least 6 characters long.");
    return;
  }

  if (password !== confirmPassword) {
    showAuthAlert("Passwords do not match. Please re-enter.");
    return;
  }

  const submitBtn = document.getElementById("registerSubmitBtn");
  const originalBtnText = submitBtn.innerHTML;
  submitBtn.disabled = true;
  submitBtn.innerHTML = "<span>Creating Account...</span>";

  try {
    const response = await fetch(REGISTER_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        fullname: fullname,
        username: username,
        email: email,
        password: password
      })
    });

    const data = await response.json().catch(() => null);

    if (!response.ok) {
      const msg = (data && data.error) ? data.error : `Registration failed (Status: ${response.status})`;
      throw new Error(msg);
    }

    // Success: log newly registered user directly into dashboard
    console.log("[Auth] Registered successfully:", data.user);
    setAuthenticatedUser(data.user);

  } catch (err) {
    console.error("[Auth] Registration error:", err);
    let errorMsg = err.message || "Failed to register account.";
    if (err.name === "TypeError" && err.message.includes("fetch")) {
      errorMsg = "Unable to connect to Flask server (http://127.0.0.1:5000). Please make sure backend is running: python app.py";
    }
    showAuthAlert(errorMsg, "error");
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = originalBtnText;
  }
}


// =============================================================================
// Machine Learning Inference & Dashboard Logic
// =============================================================================

/**
 * Switch Result Card View States
 * @param {'initial'|'loading'|'error'|'result'} state 
 */
function setResultState(state) {
  initialView.style.display = (state === 'initial') ? 'block' : 'none';
  loadingView.style.display = (state === 'loading') ? 'block' : 'none';
  errorView.style.display = (state === 'error') ? 'block' : 'none';
  resultView.style.display = (state === 'result') ? 'block' : 'none';
}

/**
 * Periodically check if the Flask ML API is online
 */
async function checkServerHealth() {
  try {
    const response = await fetch(HEALTH_ENDPOINT, { method: "GET" });
    if (response.ok) {
      apiStatusChip.className = "api-status-chip online";
      apiStatusLabel.textContent = "Flask API: Online";
      return true;
    } else {
      throw new Error(`Status ${response.status}`);
    }
  } catch (err) {
    apiStatusChip.className = "api-status-chip offline";
    apiStatusLabel.textContent = "Flask API: Offline";
    return false;
  }
}

/**
 * Populate form inputs from a preset object
 * @param {Object} profile 
 */
function applySampleProfile(profile) {
  FEATURE_KEYS.forEach(key => {
    const input = document.getElementById(key);
    if (input && profile[key] !== undefined) {
      input.value = profile[key];
      // Visual feedback highlight
      input.style.borderColor = "var(--primary-blue)";
      setTimeout(() => {
        input.style.borderColor = "";
      }, 600);
    }
  });
}

/**
 * Collect and validate all 13 clinical biomarkers from the form
 * Returns an object with numerical floats or throws an Error
 */
function extractFormData() {
  const payload = {};
  const emptyFields = [];

  FEATURE_KEYS.forEach(key => {
    const input = document.getElementById(key);
    if (!input || input.value.trim() === "") {
      emptyFields.push(key);
    } else {
      const val = parseFloat(input.value.trim());
      if (isNaN(val)) {
        emptyFields.push(key);
      } else {
        payload[key] = val;
      }
    }
  });

  if (emptyFields.length > 0) {
    throw new Error(`Please fill in all clinical fields with valid numbers. Missing or invalid: ${emptyFields.join(", ")}`);
  }

  return payload;
}

/**
 * Submit data to Flask ML backend and render prediction
 */
async function handleFormSubmit(event) {
  if (event) event.preventDefault();

  let payload;
  try {
    payload = extractFormData();
  } catch (validationErr) {
    alert(validationErr.message);
    return;
  }

  // Attach user_id if logged in
  const currentUser = getAuthenticatedUser();
  if (currentUser && currentUser.id) {
    payload.user_id = currentUser.id;
  }

  // Switch to loading state
  setResultState('loading');

  try {
    // Send POST request with JSON payload to Flask backend
    const response = await fetch(PREDICT_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json"
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const errorJson = await response.json().catch(() => null);
      const msg = (errorJson && errorJson.error) ? errorJson.error : `Server returned status ${response.status}`;
      throw new Error(msg);
    }

    const data = await response.json();
    console.log("[CardioPredict] Inference response:", data);

    // Render results received strictly from Flask Support Vector Machine model
    renderPredictionResult(data);

    // Refresh history log with the new assessment record
    fetchAndRenderHistory();

  } catch (err) {
    console.error("[CardioPredict] Error submitting prediction:", err);
    errorTitle.textContent = "Prediction Request Failed";
    errorMessageText.textContent = err.message || "Failed to communicate with Flask backend API.";
    setResultState('error');
  }
}

/**
 * Display the returned predictions on the Result Card
 * @param {Object} data - Prediction payload from Flask
 */
function renderPredictionResult(data) {
  const { prediction, result, risk, no_disease_probability, disease_probability } = data;

  const isLowRisk = (prediction === 0);

  // 1. Outcome badge styling & icons
  if (isLowRisk) {
    outcomeBox.className = "outcome-box low-risk";
    outcomeIconCircle.textContent = "✓";
    outcomeRiskBadge.textContent = risk || "Lower Risk";
    outcomeHeading.textContent = result || "No Heart Disease";
    
    clinicalAdviceText.textContent = 
      "The Support Vector Machine (SVM) model computed a high probability of healthy coronary indicators. Maintain a balanced diet, exercise routine, and regular wellness visits.";
  } else {
    outcomeBox.className = "outcome-box high-risk";
    outcomeIconCircle.textContent = "!";
    outcomeRiskBadge.textContent = risk || "Higher Risk";
    outcomeHeading.textContent = result || "Heart Disease Detected";
    
    clinicalAdviceText.textContent = 
      "Biomarkers reflect elevated likelihood of coronary artery narrowing. Follow-up consultation, ECG monitoring, and coronary angiography are advised by cardiology guidelines.";
  }

  // 2. Progress bars animation
  const noDiseaseVal = Math.max(0, Math.min(100, no_disease_probability));
  const diseaseVal = Math.max(0, Math.min(100, disease_probability));

  noDiseasePercent.textContent = `${noDiseaseVal.toFixed(1)}%`;
  diseasePercent.textContent = `${diseaseVal.toFixed(1)}%`;

  // Animate width transition
  noDiseaseProgress.style.width = "0%";
  diseaseProgress.style.width = "0%";

  setTimeout(() => {
    noDiseaseProgress.style.width = `${noDiseaseVal}%`;
    diseaseProgress.style.width = `${diseaseVal}%`;
  }, 50);

  // 3. Mini summary tiles
  rawClassVal.textContent = isLowRisk ? "Class 0 (Negative)" : "Class 1 (Positive)";
  const confidence = isLowRisk ? noDiseaseVal : diseaseVal;
  modelConfidenceVal.textContent = `${confidence.toFixed(1)}%`;
  riskCategoryVal.textContent = risk;

  // Show result state
  setResultState('result');
}

/**
 * Tab Navigation Controller
 * @param {string} targetTabId 
 */
function switchNavTab(targetTabId) {
  if (targetTabId === "biomarkers-link") {
    // Show predict tab and smooth-scroll down to the embedded Biomarkers Guide
    switchNavTab("predict-tab");
    setTimeout(() => {
      const guideEl = document.getElementById("biomarkers-guide-section");
      if (guideEl) {
        guideEl.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }, 50);

    // Keep Biomarkers Guide button highlighted in sidebar
    document.querySelectorAll(".nav-btn").forEach(btn => {
      if (btn.getAttribute("data-tab") === "biomarkers-link") {
        btn.classList.add("active");
      }
    });
    return;
  }

  // Update nav buttons
  document.querySelectorAll(".nav-btn").forEach(btn => {
    if (btn.getAttribute("data-tab") === targetTabId) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });

  // Update tab panels
  document.querySelectorAll(".tab-panel").forEach(panel => {
    if (panel.id === targetTabId) {
      panel.style.display = "block";
    } else {
      panel.style.display = "none";
    }
  });
}


// =============================================================================
// Initialization and Event Listeners
// =============================================================================

document.addEventListener("DOMContentLoaded", () => {
  // 1. Initialize Authentication State
  checkExistingSession();

  // 2. Auth Tab Buttons & Switch Links
  if (tabLoginBtn) {
    tabLoginBtn.addEventListener("click", () => switchAuthMode("login"));
  }
  if (tabRegisterBtn) {
    tabRegisterBtn.addEventListener("click", () => switchAuthMode("register"));
  }
  if (linkToRegister) {
    linkToRegister.addEventListener("click", (e) => {
      e.preventDefault();
      switchAuthMode("register");
    });
  }
  if (linkToLogin) {
    linkToLogin.addEventListener("click", (e) => {
      e.preventDefault();
      switchAuthMode("login");
    });
  }

  // 3. One-Click Quick Demo Login Button
  if (quickDemoBtn) {
    quickDemoBtn.addEventListener("click", () => {
      document.getElementById("loginIdentifier").value = "demo";
      document.getElementById("loginPassword").value = "demo123";
      loginForm.dispatchEvent(new Event("submit", { cancelable: true }));
    });
  }

  // 4. Form Submissions
  if (loginForm) {
    loginForm.addEventListener("submit", handleLoginSubmit);
  }
  if (registerForm) {
    registerForm.addEventListener("submit", handleRegisterSubmit);
  }

  // 5. Logout Button
  if (logoutBtn) {
    logoutBtn.addEventListener("click", logoutUser);
  }

  // 6. Navigation Tabs
  document.querySelectorAll(".nav-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const tabId = btn.getAttribute("data-tab");
      switchNavTab(tabId);
    });
  });

  // 7. Prediction Form Submission & Reset
  if (heartForm) {
    heartForm.addEventListener("submit", handleFormSubmit);
  }
  if (resetFormBtn) {
    resetFormBtn.addEventListener("click", () => {
      heartForm.reset();
      setResultState('initial');
    });
  }

  // 8. Preset Sample Buttons
  if (loadHealthySampleBtn) {
    loadHealthySampleBtn.addEventListener("click", () => {
      applySampleProfile(SAMPLE_PROFILES.healthy);
    });
  }
  if (loadAtRiskSampleBtn) {
    loadAtRiskSampleBtn.addEventListener("click", () => {
      applySampleProfile(SAMPLE_PROFILES.atRisk);
    });
  }

  // 9. Scroll to 13 Parameters Guide button
  if (scrollToGuideBtn) {
    scrollToGuideBtn.addEventListener("click", () => {
      const guideEl = document.getElementById("biomarkers-guide-section");
      if (guideEl) {
        guideEl.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  }

  // 9b. Jump to Model Metrics Tab button
  const jumpToMetricsBtn = document.getElementById("jumpToMetricsBtn");
  if (jumpToMetricsBtn) {
    jumpToMetricsBtn.addEventListener("click", () => {
      switchNavTab("metrics-tab");
    });
  }

  // 9c. Jump to History Tab button
  const jumpToHistoryBtn = document.getElementById("jumpToHistoryBtn");
  if (jumpToHistoryBtn) {
    jumpToHistoryBtn.addEventListener("click", () => {
      switchNavTab("history-tab");
    });
  }

  // 9d. History Controls (Refresh, Clear, Search, Filter)
  const refreshHistoryBtn = document.getElementById("refreshHistoryBtn");
  if (refreshHistoryBtn) {
    refreshHistoryBtn.addEventListener("click", fetchAndRenderHistory);
  }

  const clearHistoryBtn = document.getElementById("clearHistoryBtn");
  if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener("click", clearAllHistory);
  }

  const historySearchInput = document.getElementById("historySearchInput");
  if (historySearchInput) {
    historySearchInput.addEventListener("input", renderHistoryTable);
  }

  const historyRiskFilter = document.getElementById("historyRiskFilter");
  if (historyRiskFilter) {
    historyRiskFilter.addEventListener("change", renderHistoryTable);
  }

  // 10. Retry button
  if (errorRetryBtn) {
    errorRetryBtn.addEventListener("click", () => {
      handleFormSubmit();
    });
  }

  // 11. Initial health check and periodic heartbeat every 15s
  checkServerHealth();
  setInterval(checkServerHealth, 15000);
});


// =============================================================================
// Patient Assessment History Management
// =============================================================================

let cachedHistory = [];

/**
 * Fetch patient prediction history records from Flask backend
 */
async function fetchAndRenderHistory() {
  const user = getAuthenticatedUser();
  const url = user && user.id ? `${HISTORY_ENDPOINT}?user_id=${user.id}` : HISTORY_ENDPOINT;

  try {
    const res = await fetch(url, { method: "GET" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    cachedHistory = data.history || [];
    updateHistoryStats();
    renderHistoryTable();
  } catch (err) {
    console.warn("[History] Failed to fetch history records:", err);
  }
}

/**
 * Update KPI counts and navigation pill badge
 */
function updateHistoryStats() {
  const total = cachedHistory.length;
  const highRisk = cachedHistory.filter(h => h.prediction === 1).length;
  const lowRisk = cachedHistory.filter(h => h.prediction === 0).length;

  const totalEl = document.getElementById("totalHistoryCount");
  const highEl = document.getElementById("highRiskHistoryCount");
  const lowEl = document.getElementById("lowRiskHistoryCount");
  const badgeEl = document.getElementById("historyNavBadge");

  if (totalEl) totalEl.textContent = total;
  if (highEl) highEl.textContent = highRisk;
  if (lowEl) lowEl.textContent = lowRisk;
  if (badgeEl) badgeEl.textContent = total;
}

/**
 * Render history table rows with search and filter capabilities
 */
function renderHistoryTable() {
  const tbody = document.getElementById("historyTableBody");
  const emptyState = document.getElementById("historyEmptyState");
  if (!tbody) return;

  const searchVal = (document.getElementById("historySearchInput")?.value || "").toLowerCase().trim();
  const riskFilter = document.getElementById("historyRiskFilter")?.value || "ALL";

  let filtered = cachedHistory.filter(item => {
    // Risk filter
    if (riskFilter === "HIGHER" && item.prediction !== 1) return false;
    if (riskFilter === "LOWER" && item.prediction !== 0) return false;

    // Search query filter
    if (searchVal) {
      const matchName = (item.patient_name || "").toLowerCase().includes(searchVal);
      const matchAge = String(item.age).includes(searchVal);
      const matchResult = (item.result || "").toLowerCase().includes(searchVal);
      return matchName || matchAge || matchResult;
    }
    return true;
  });

  if (filtered.length === 0) {
    tbody.innerHTML = "";
    if (emptyState) emptyState.style.display = "block";
    return;
  }

  if (emptyState) emptyState.style.display = "none";

  tbody.innerHTML = filtered.map(item => {
    const isHigh = (item.prediction === 1);
    const badgeClass = isHigh ? "risk-badge-high" : "risk-badge-low";
    const badgeIcon = isHigh ? "⚠️" : "✓";
    const riskLabel = isHigh ? "Higher Risk" : "Lower Risk";
    const probVal = isHigh ? item.disease_probability : item.no_disease_probability;
    const gender = item.sex === 1 ? "Male" : "Female";

    // Format human-friendly timestamp converted accurately to user's local timezone
    let formattedDate = item.created_at || "Recent";
    try {
      let rawStr = (item.created_at || "").trim();
      if (rawStr) {
        // If it's a UTC SQLite timestamp without timezone, append Z for proper local conversion
        if (!rawStr.includes("Z") && !rawStr.includes("+")) {
          rawStr = rawStr.replace(" ", "T") + "Z";
        }
        const d = new Date(rawStr);
        if (!isNaN(d.getTime())) {
          formattedDate = d.toLocaleDateString(undefined, { 
            month: 'short', day: 'numeric' 
          }) + ", " + d.toLocaleTimeString(undefined, {
            hour: '2-digit', minute: '2-digit', hour12: true
          });
        }
      }
    } catch(e) {}

    return `
      <tr data-history-id="${item.id}">
        <td><strong>#${item.id}</strong></td>
        <td>
          <span class="hist-patient-name">${escapeHtml(item.patient_name || 'Patient')}</span>
          <span class="hist-patient-meta">${item.age} yrs &bull; ${gender}</span>
        </td>
        <td>
          <div class="hist-vitals-pill">
            <span>BP: <strong>${item.trestbps}</strong></span>
            <span>Chol: <strong>${item.chol}</strong></span>
            <span>MaxHR: <strong>${item.thalach}</strong></span>
          </div>
        </td>
        <td>
          <span class="hist-risk-badge ${badgeClass}">${badgeIcon} ${riskLabel}</span>
          <small style="display:block; margin-top:3px; color: var(--text-muted); font-size: 0.72rem;">
            Confidence: ${probVal}%
          </small>
        </td>
        <td>
          <span class="hist-time-text">${formattedDate}</span>
        </td>
        <td>
          <div class="hist-actions-group">
            <button type="button" class="btn-table-action btn-table-load" onclick="loadHistoryItemIntoForm(${item.id})" title="Load these parameters into form">
              Load In Form
            </button>
            <button type="button" class="btn-table-action btn-table-del" onclick="deleteHistoryRecord(${item.id})" title="Delete record">
              🗑️
            </button>
          </div>
        </td>
      </tr>
    `;
  }).join("");
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

/**
 * Load a past patient's 13 biomarkers back into the prediction form
 */
function loadHistoryItemIntoForm(historyId) {
  const item = cachedHistory.find(h => h.id === historyId);
  if (!item) return;

  applySampleProfile({
    age: item.age,
    sex: String(parseInt(item.sex)),
    cp: String(parseInt(item.cp)),
    trestbps: item.trestbps,
    chol: item.chol,
    fbs: String(parseInt(item.fbs)),
    restecg: String(parseInt(item.restecg)),
    thalach: item.thalach,
    exang: String(parseInt(item.exang)),
    oldpeak: item.oldpeak,
    slope: String(parseInt(item.slope)),
    ca: String(parseInt(item.ca)),
    thal: String(parseInt(item.thal))
  });

  // Switch to predict tab and scroll to form
  switchNavTab("predict-tab");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/**
 * Delete a single record from database
 */
async function deleteHistoryRecord(historyId) {
  if (!confirm(`Are you sure you want to delete patient record #${historyId}?`)) return;

  try {
    const res = await fetch(`${HISTORY_ENDPOINT}/${historyId}`, { method: "DELETE" });
    if (res.ok) {
      await fetchAndRenderHistory();
    }
  } catch (err) {
    alert("Failed to delete record: " + err.message);
  }
}

/**
 * Clear all history records
 */
async function clearAllHistory() {
  if (!confirm("Are you sure you want to clear all patient prediction history? This action cannot be undone.")) return;

  const user = getAuthenticatedUser();
  try {
    const res = await fetch(`${HISTORY_ENDPOINT}/clear`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: user ? user.id : null })
    });
    if (res.ok) {
      await fetchAndRenderHistory();
    }
  } catch (err) {
    alert("Failed to clear history: " + err.message);
  }
}

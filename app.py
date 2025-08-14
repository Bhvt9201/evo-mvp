# app.py  (FULL replacement - includes model loading, labeling, retrain)
import time, uuid, json, os, random, math, threading, subprocess, sys
from collections import defaultdict
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import numpy as np
import joblib

# Config files
PERSIST_FILE = "persist.json"
LABELS_FILE = "labels.json"
MODEL_FILE = "model.pkl"
MODEL_META = "model_meta.json"

INTENT_LABELS = ["purchase", "research", "browse"]

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# In-memory stores (MVP)
SESSIONS = {}
EVENT_LOG = []
DECISIONS = []
BANDIT = {}

# Model holder
MODEL = None
MODEL_LOCK = threading.Lock()
TRAINING_THREAD = None

# -------- persistence helpers ----------
def load_persist():
    if os.path.exists(PERSIST_FILE):
        try:
            with open(PERSIST_FILE, "r") as f:
                data = json.load(f)
                SESSIONS.update(data.get("sessions", {}))
                EVENT_LOG.extend(data.get("events", [])[-500:])
                DECISIONS.extend(data.get("decisions", [])[-500:])
                print("Loaded persisted state.")
        except Exception as e:
            print("Persist load error:", e)

def persist():
    try:
        with open(PERSIST_FILE, "w") as f:
            json.dump({"sessions": SESSIONS, "events": EVENT_LOG[-400:], "decisions": DECISIONS[-400:]}, f)
    except Exception as e:
        print("Persist error:", e)

load_persist()

# -------- bandit utilities ----------
def create_bandit(exp_id, arms):
    if exp_id not in BANDIT:
        BANDIT[exp_id] = {"arms": arms, "alpha":[1]*len(arms), "beta":[1]*len(arms), "trials":[0]*len(arms), "wins":[0]*len(arms)}
    return BANDIT[exp_id]

def choose_bandit_arm(exp_id):
    b = BANDIT.get(exp_id)
    if not b: return 0, b["arms"][0] if b else None
    samples = [np.random.beta(a,b_) for a,b_ in zip(b["alpha"], b["beta"])]
    idx = int(np.argmax(samples))
    return idx, b["arms"][idx]

def update_bandit(exp_id, arm_idx, reward):
    b = BANDIT.get(exp_id)
    if not b: return
    b["trials"][arm_idx] += 1
    if reward:
        b["wins"][arm_idx] += 1
        b["alpha"][arm_idx] += 1
    else:
        b["beta"][arm_idx] += 1

create_bandit("cta_v1", ["Buy now — 1 click", "Get it today", "Shop now"])

# -------- model loading & prediction ----------
def load_model():
    global MODEL
    if os.path.exists(MODEL_FILE):
        try:
            m = joblib.load(MODEL_FILE)
            with MODEL_LOCK:
                MODEL = m
            print("Loaded model from", MODEL_FILE)
            return True
        except Exception as e:
            print("Failed to load model:", e)
    print("No model file found; using synthetic fallback.")
    MODEL = None
    return False

load_model()

def extract_features_from_payload(payload, session_summary):
    clicks = session_summary.get("clicks", 0)
    scroll = float(session_summary.get("scroll_max", 0.0))
    mouse = session_summary.get("mouse_moves", 0)
    ref = (payload.get("referrer") or "").lower()
    hour = payload.get("local_hour", 12)
    is_search = 1 if any(k in ref for k in ["google", "bing", "search"]) else 0
    is_social = 1 if any(k in ref for k in ["reddit", "twitter", "instagram", "facebook"]) else 0
    return np.array([[clicks, scroll, is_search, is_social, hour/24.0, mouse]])

def predict_intent(payload, session_summary):
    features = extract_features_from_payload(payload, session_summary)
    with MODEL_LOCK:
        model = MODEL
    if model is not None:
        try:
            probs = model.predict_proba(features)[0]
            idx = int(np.argmax(probs))
            return INTENT_LABELS[idx], float(probs[idx])
        except Exception as e:
            print("Model prediction error:", e)
    # fallback synthetic heuristics
    score = {"purchase":0.0, "research":0.0, "browse":0.0}
    ref = (payload.get("referrer") or "").lower()
    if "google" in ref or "search" in ref:
        score["purchase"] += 0.6
    if "reddit" in ref or "twitter" in ref or "instagram" in ref:
        score["browse"] += 0.5
    clicks = session_summary.get("clicks", 0)
    scroll = session_summary.get("scroll_max", 0)
    if clicks >= 2 and scroll < 0.25:
        score["purchase"] += 0.45
    if scroll > 0.5:
        score["research"] += 0.45
    long_hover = False
    try:
        hovers = (payload.get("first_seconds") or {}).get("hovers", {})
        long_hover = any(v.get("total",0) > 500 for v in hovers.values())
    except:
        pass
    if long_hover:
        score["browse"] += 0.15
    hour = payload.get("local_hour", 12)
    if hour >= 22 or hour <= 6:
        score["browse"] += 0.12
    intent = max(score.items(), key=lambda x: x[1])[0]
    confidence = float(score[intent])
    return intent, confidence

# -------- training utilities (background) ----------
def run_training_background():
    global TRAINING_THREAD
    if TRAINING_THREAD and TRAINING_THREAD.is_alive():
        return False, "Training already running"
    TRAINING_THREAD = threading.Thread(target=do_train_and_reload, daemon=True)
    TRAINING_THREAD.start()
    return True, "Training started"

def do_train_and_reload():
    try:
        print("Starting background training using train_intent.py ...")
        subprocess.run([sys.executable, "train_intent.py"], check=True)
        loaded = load_model()
        print("Background training finished. Model loaded:", loaded)
    except Exception as e:
        print("Background training failed:", e)

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/telemetry", methods=["POST"])
def telemetry():
    ev = request.get_json(force=True)
    ev["_received_at"] = time.time()
    sid = ev.get("session_id") or f"anon_{uuid.uuid4().hex[:8]}"
    ev["session_id"] = sid
    EVENT_LOG.append(ev)
    if len(EVENT_LOG) > 1000:
        del EVENT_LOG[:-800]
    sess = SESSIONS.setdefault(sid, {"session_id": sid, "first_seen": ev["_received_at"], "events": [], "summary": {"clicks":0,"scroll_max":0.0,"mouse_moves":0}})
    sess["events"].append(ev)
    if len(sess["events"]) > 500:
        del sess["events"][:-400]
    meta = ev.get("interaction", {})
    sess["summary"]["clicks"] += meta.get("clicks", 0)
    sess["summary"]["scroll_max"] = max(sess["summary"]["scroll_max"], meta.get("scroll_depth", 0))
    sess["summary"]["mouse_moves"] += meta.get("mouse_moves", 0)
    if int(time.time()) % 17 == 0:
        persist()
    return jsonify({"ok": True, "session_id": sid})

@app.route("/api/decide", methods=["POST"])
def decide():
    payload = request.get_json(force=True) or {}
    sid = payload.get("session_id")
    sess = SESSIONS.get(sid, {"session_id": sid, "summary": {"clicks":0,"scroll_max":0.0,"mouse_moves":0}})
    intent, confidence = predict_intent(payload, sess.get("summary", {}))
    arm_idx, cta_text = choose_bandit_arm("cta_v1")
    if intent == "purchase" and confidence >= 0.45:
        action = "show_quick_checkout"
        design = {"theme":"compact", "cta_text": cta_text, "accent":"#0a84ff"}
    elif intent == "research":
        action = "highlight_compare"
        design = {"theme":"informative", "cta_text": cta_text, "accent":"#ff8c00"}
    else:
        action = "show_default"
        design = {"theme":"friendly", "cta_text": cta_text, "accent":"#6c6cff"}
    explanation = []
    ref = (payload.get("referrer") or "").lower()
    if "google" in ref or "search" in ref:
        explanation.append("Arrived from search (strong purchase signal)")
    if "instagram" in ref or "reddit" in ref:
        explanation.append("Social referrer (browsing)")
    first = payload.get("first_seconds", {})
    if first.get("clicks",0) >= 2 and first.get("scroll_depth",0) < 0.25:
        explanation.append("Multiple clicks + low scroll — likely ready to buy")
    if first.get("scroll_depth",0) > 0.5:
        explanation.append("Deep scrolling — researching details")
    hovers = first.get("hovers", {}) if isinstance(first, dict) else {}
    if any(v.get("total",0) > 500 for v in hovers.values()):
        explanation.append("Long hover on visuals — visual explorer")
    explanation_text = "; ".join(explanation) if explanation else "Combined passive signals"
    decision = {"t": time.time(), "session_id": sid, "intent": intent, "confidence": round(confidence,2), "action": action, "design": design, "explanation": explanation_text, "chosen_arm_idx": arm_idx}
    DECISIONS.append(decision)
    if len(DECISIONS) > 1000:
        del DECISIONS[:-800]
    return jsonify(decision)

@app.route("/api/outcome", methods=["POST"])
def outcome():
    payload = request.get_json(force=True) or {}
    sid = payload.get("session_id")
    reward = bool(payload.get("reward", False))
    last = None
    for d in reversed(DECISIONS):
        if d.get("session_id") == sid:
            last = d
            break
    if last:
        exp_id = "cta_v1"
        update_bandit(exp_id, last["chosen_arm_idx"], 1 if reward else 0)
    labels = {}
    if os.path.exists(LABELS_FILE):
        try:
            with open(LABELS_FILE, "r") as f:
                labels = json.load(f)
        except:
            labels = {}
    labels.setdefault(sid, {}).update({"outcome_label": "converted" if reward else "no_conversion", "outcome_at": time.time()})
    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f)
    return jsonify({"ok": True})

@app.route("/api/label", methods=["POST"])
def api_label():
    payload = request.get_json(force=True) or {}
    sid = payload.get("session_id")
    intent = payload.get("intent")
    annotator = payload.get("annotator", "admin")
    if intent not in INTENT_LABELS:
        return jsonify({"ok": False, "error": "invalid_intent"}), 400
    labels = {}
    if os.path.exists(LABELS_FILE):
        try:
            with open(LABELS_FILE, "r") as f:
                labels = json.load(f)
        except:
            labels = {}
    labels[sid] = {"intent": intent, "labelled_at": time.time(), "annotator": annotator}
    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f)
    return jsonify({"ok": True})

@app.route("/admin")
def admin():
    return render_template("admin.html")

@app.route("/admin/sessions")
def admin_sessions():
    s = []
    for sid, v in list(SESSIONS.items())[-400:]:
        s.append({"session_id": sid, "first_seen": v.get("first_seen"), "summary": v.get("summary", {}), "has_label": _has_label(sid)})
    return jsonify({"count": len(SESSIONS), "sessions": s})

def _has_label(session_id):
    if os.path.exists(LABELS_FILE):
        try:
            with open(LABELS_FILE, "r") as f:
                labels = json.load(f)
                return session_id in labels
        except:
            return False
    return False

@app.route("/admin/events")
def admin_events():
    return jsonify({"recent_events": EVENT_LOG[-200:]})

@app.route("/admin/decisions")
def admin_decisions():
    return jsonify({"recent_decisions": DECISIONS[-200:], "bandit": BANDIT})

@app.route("/admin/labels")
def admin_labels():
    labels = {}
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            labels = json.load(f)
    return jsonify(labels)

@app.route("/admin/retrain", methods=["POST"])
def admin_retrain():
    ok, msg = run_training_background()
    return jsonify({"ok": ok, "msg": msg})

if __name__ == "__main__":
    print("Starting Evo MVP backend on http://0.0.0.0:5001")
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))

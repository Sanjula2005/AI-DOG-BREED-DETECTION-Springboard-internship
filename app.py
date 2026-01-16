

import json
import torch
import timm
import psycopg2
import datetime
import jwt

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torchvision.transforms as T
import google.generativeai as genai

from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# ==========================================================
# CONFIG
# ==========================================================
MODEL_PATH = "model/dog_breed_model_1.pth"
CLASSES_PATH = "model/classes_extracted.txt"
BREED_INFO_PATH = "data/120_breeds_new.json"

JWT_SECRET = "CHANGE_THIS_SECRET_KEY"
JWT_ALGO = "HS256"

genai.configure(api_key="Gemini_api_key")

# ==========================================================
# APP INIT
# ==========================================================
app = Flask(__name__)
CORS(app)

device = torch.device("cpu")

# ==========================================================
# DATABASE
# ==========================================================
conn = psycopg2.connect(
    host="localhost",
    port="5432",
    database="dog_app",
    user="postgres",
    password="your_password"
)
conn.autocommit = True
cursor = conn.cursor()

# ==========================================================
# LOAD CLASSES & BREED INFO
# ==========================================================
with open(CLASSES_PATH) as f:
    class_names = [c.strip() for c in f.readlines()]

num_classes = len(class_names)

with open(BREED_INFO_PATH) as f:
    raw_info = json.load(f)

breed_info = {b["Breed"].lower(): b for b in raw_info}

# ==========================================================
# JWT HELPERS
# ==========================================================
def generate_token(user_id):
    payload = {
        "user_id": user_id,
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=6)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)

def token_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not token:
            return jsonify({"error": "Missing token"}), 401
        try:
            decoded = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
            request.user_id = decoded["user_id"]
        except:
            return jsonify({"error": "Invalid or expired token"}), 401
        return f(*args, **kwargs)
    return wrapper

# ==========================================================
# AUTH
# ==========================================================
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    hashed = generate_password_hash(data["password"])
    cursor.execute(
        "INSERT INTO users (username, password) VALUES (%s,%s)",
        (data["username"], hashed)
    )
    return jsonify({"message": "Registered successfully"})

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    cursor.execute(
        "SELECT id, password FROM users WHERE username=%s",
        (data["username"],)
    )
    user = cursor.fetchone()
    if not user or not check_password_hash(user[1], data["password"]):
        return jsonify({"error": "Invalid credentials"}), 401
    return jsonify({
        "token": generate_token(user[0]),
        "user_id": user[0]
    })

# ==========================================================
# MODEL LOAD
# ==========================================================
model = None

def load_model():
    global model
    if model is None:
        m = timm.create_model(
            "mobilenetv3_large_100",
            pretrained=False,
            num_classes=num_classes
        )
        ckpt = torch.load(MODEL_PATH, map_location=device)
        m.load_state_dict(ckpt.get("state", ckpt))
        m.eval()
        model = m

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor()
])

def preprocess(img):
    return transform(img.convert("RGB")).unsqueeze(0)

# ==========================================================
# PREDICT
# ==========================================================
@app.route("/predict", methods=["POST"])
@token_required
def predict():
    load_model()

    img = Image.open(request.files["file"])
    x = preprocess(img)

    with torch.no_grad():
        probs = torch.softmax(model(x)[0], dim=0)

    vals, idxs = torch.topk(probs, 5)
    top5 = [{
        "breed": class_names[i],
        "confidence": round(float(v*100), 2)
    } for v, i in zip(vals, idxs)]

    top1 = top5[0]

    cursor.execute(
        "INSERT INTO predictions (user_id, breed, confidence) VALUES (%s,%s,%s)",
        (request.user_id, top1["breed"], top1["confidence"])
    )

    cursor.execute("""
        INSERT INTO chat_sessions (user_id, last_breed)
        VALUES (%s,%s)
        ON CONFLICT (user_id)
        DO UPDATE SET last_breed=EXCLUDED.last_breed, updated_at=NOW()
    """, (request.user_id, top1["breed"]))

    return jsonify({"predicted": top1, "top5": top5})

# ==========================================================
# HISTORY
# ==========================================================
@app.route("/history")
@token_required
def history():
    cursor.execute("""
        SELECT breed, confidence, timestamp
        FROM predictions
        WHERE user_id=%s
        ORDER BY timestamp DESC
    """, (request.user_id,))
    return jsonify([
        {"breed": b, "confidence": c, "timestamp": t.isoformat()}
        for b,c,t in cursor.fetchall()
    ])

# ==========================================================
# CHATBOT (SESSION AWARE)
# ==========================================================
def is_dog_related(q):
    prompt = f"""
    Is the question about dogs, dog breeds, diet, health or care?
    Question: "{q}"
    Answer only YES or NO
    """
    r = genai.GenerativeModel("gemini-2.5-flash").generate_content(
        prompt, generation_config={"temperature": 0}
    )
    return r.text.strip().upper() == "YES"

@app.route("/chat", methods=["POST"])
@token_required
def chat():
    user_input = request.form.get("user_input")
    if not user_input:
        return jsonify({"error": "Missing user_input"}), 400

    cursor.execute(
        "SELECT last_breed FROM chat_sessions WHERE user_id=%s",
        (request.user_id,)
    )
    row = cursor.fetchone()
    breed = row[0].lower() if row else "general dog"

    if not is_dog_related(user_input):
        return jsonify({"reply": "I can only answer dog-related questions."})

    cursor.execute("""
        SELECT role, message FROM chat_history
        WHERE user_id=%s
        ORDER BY timestamp DESC
        LIMIT 6
    """, (request.user_id,))
    history = cursor.fetchall()[::-1]

    context = "\n".join(f"{r}: {m}" for r,m in history)

    prompt = f"""
    You are a veterinary expert.

    Breed context: {breed}

    Conversation:
    {context}

    User: {user_input}
    """

    reply = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt).text

    cursor.execute(
        "INSERT INTO chat_history (user_id, role, message) VALUES (%s,'user',%s)",
        (request.user_id, user_input)
    )
    cursor.execute(
        "INSERT INTO chat_history (user_id, role, message) VALUES (%s,'bot',%s)",
        (request.user_id, reply)
    )

    return jsonify({"reply": reply})

# ==========================================================
# MAP
# ==========================================================
@app.route("/map-redirect")
@token_required
def map_redirect():
    loc = request.args.get("location")
    svc = request.args.get("service")
    return jsonify({
        "url": f"https://www.google.com/maps/search/{svc}+near+{loc}"
    })



@app.route("/breed-info", methods=["GET"])
@token_required
def breed_info_api():
    breed = request.args.get("breed")

    if not breed:
        cursor.execute(
            "SELECT last_breed FROM chat_sessions WHERE user_id=%s",
            (request.user_id,)
        )
        row = cursor.fetchone()
        breed = row[0] if row else None

    if not breed:
        return jsonify({"error": "Breed not found"}), 404

    breed = breed.lower()

    if breed not in breed_info:
        return jsonify({"error": "Breed not found"}), 404

    return jsonify(breed_info[breed])



@app.route("/diet-plan", methods=["GET"])
@token_required
def get_diet_plan():
    stage = request.args.get("stage", "")

    if not stage:
        return jsonify({"error": "stage required"}), 400

    with open("data/general_diet_plan.json") as f:
        data = json.load(f)

    if stage not in data:
        return jsonify({"error": "Invalid stage"}), 400

    plan = data[stage]

    # 🔥 normalize keys
    normalized = {k.lower(): v for k, v in plan.items()}

    return jsonify(normalized)
# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":
    app.run(debug=True)



# # import os
# # import json
# # import torch
# # import timm
# # from flask import Flask, request, jsonify
# # from PIL import Image
# # import torchvision.transforms as T
# # import google.generativeai as genai

# # # -------------------------------------------
# # # CONFIG
# # # -------------------------------------------
# # MODEL_PATH = "model/dog_breed_model_1.pth"     
# # CLASSES_PATH = "model/classes_extracted.txt"   
# # BREED_INFO_PATH = "data/120_breeds_new.json"

# # device = torch.device("cpu")
# # app = Flask(__name__)

# # # -------------------------------------------
# # # LOAD CLASS NAMES
# # # -------------------------------------------
# # with open(CLASSES_PATH, "r") as f:
# #     class_names = [line.strip() for line in f.readlines()]

# # num_classes = len(class_names)

# # # -------------------------------------------
# # # LOAD BREED INFO
# # # -------------------------------------------
# # with open(BREED_INFO_PATH, "r") as f:
# #     raw_info = json.load(f)

# # breed_info = {item["Breed"].lower(): item for item in raw_info}

# # # -------------------------------------------
# # # LAZY MODEL LOAD
# # # -------------------------------------------
# # model = None

# # def load_model():
# #     global model
# #     if model is None:
# #         print("Loading model…")

# #         m = timm.create_model(
# #             "mobilenetv3_large_100",
# #             pretrained=False,
# #             num_classes=num_classes
# #         )

# #         ckpt = torch.load(MODEL_PATH, map_location=device)
# #         state = ckpt.get("state", ckpt)
# #         m.load_state_dict(state)

# #         m.to(device)
# #         m.eval()

# #         model = m
# #         print("Model loaded successfully!")


# # preprocess = T.Compose([
# #     T.Resize((224, 224)),
# #     T.ToTensor()      
# # ])

# # def preprocess_image(image):
# #     image = image.convert("RGB")
# #     tensor = preprocess(image)
# #     return tensor.unsqueeze(0).to(device)

# # # ===========================================================
# # # 1️⃣ PREDICT — returns top-1 & top-5
# # # ===========================================================
# # @app.route("/predict", methods=["POST"])
# # def predict():
# #     load_model()

# #     if "file" not in request.files:
# #         return jsonify({"error": "No image uploaded"}), 400

# #     try:
# #         image = Image.open(request.files["file"])
# #     except Exception as e:
# #         return jsonify({"error": f"Invalid image: {e}"}), 400

# #     tensor = preprocess_image(image)

# #     with torch.no_grad():
# #         outputs = model(tensor)
# #         probs = torch.softmax(outputs, dim=1)[0]

# #     top_vals, top_idx = torch.topk(probs, 5)

# #     top5 = []
# #     for val, idx in zip(top_vals, top_idx):
# #         breed = class_names[idx.item()]
# #         conf = round(float(val.item() * 100), 2)
# #         top5.append({"breed": breed, "confidence": conf})

# #     return jsonify({
# #         "predicted": top5[0],
# #         "top5": top5
# #     })

# # # ===========================================================
# # # 2️⃣ BREED INFO
# # # ===========================================================
# # @app.route("/breed-info", methods=["GET"])
# # def breed_info_api():
# #     breed = request.args.get("breed", "").lower()
# #     if not breed:
# #         return jsonify({"error": "Missing breed parameter"}), 400
# #     if breed not in breed_info:
# #         return jsonify({"error": "Breed not found"}), 404
# #     return jsonify(breed_info[breed])

# # # ===========================================================
# # # 3️⃣ CHATBOT
# # # ===========================================================
# # genai.configure(api_key="AIzaSyDno2FRb6ObaQtGgGm_mf09VLnTI4CgATM")

# # @app.route("/chat", methods=["POST"])
# # def chat():
# #     breed = request.form.get("breed", "").lower()
# #     user_input = request.form.get("user_input", "")

# #     if not breed or not user_input:
# #         return jsonify({"error": "Missing breed or user_input"}), 400

# #     # Rule-based info
# #     if breed in breed_info:
# #         info = breed_info[breed]
# #         for key, value in info.items():
# #             if key.lower() in user_input.lower():
# #                 return jsonify({"reply": f"{key}: {value}"})
# #         if any(term in user_input.lower() for term in ["info", "details", "about"]):
# #             return jsonify({"reply": json.dumps(info, indent=2)})

# #     # Gemini fallback
# #     prompt = f"""
# #     The user is asking about {breed}.
# #     Question: {user_input}
# #     Provide an accurate response.
# #     """

# #     gem = genai.GenerativeModel("gemini-2.5-flash")
# #     reply = gem.generate_content(prompt).text

# #     return jsonify({"reply": reply})

# # # -------------------------------------------
# # # RUN SERVER
# # # -------------------------------------------
# # if __name__ == "__main__":
# #     print("Server running at http://127.0.0.1:5000")
# #     app.run(host="127.0.0.1", port=5000, debug=True)



# import os
# import json
# import torch
# import timm
# import psycopg2
# import datetime
# import jwt

# from flask import Flask, request, jsonify
# from PIL import Image
# import torchvision.transforms as T
# import google.generativeai as genai
# from werkzeug.security import generate_password_hash, check_password_hash
# from functools import wraps

# # -------------------------------------------
# # CONFIG
# # -------------------------------------------
# MODEL_PATH = "model/dog_breed_model_1.pth"
# CLASSES_PATH = "model/classes_extracted.txt"
# BREED_INFO_PATH = "data/120_breeds_new.json"
# # DIET_PLAN_PATH = "data/120_breeds_diet_plans.json"

# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# device = torch.device("cpu")
# # app = Flask(__name__)

# # -------------------------------------------
# # LOAD CLASS NAMES
# # -------------------------------------------
# with open(CLASSES_PATH, "r") as f:
#     class_names = [line.strip() for line in f.readlines()]

# num_classes = len(class_names)

# # -------------------------------------------
# # LOAD BREED INFO
# # -------------------------------------------
# with open(BREED_INFO_PATH, "r") as f:
#     raw_info = json.load(f)

# breed_info = {item["Breed"].lower(): item for item in raw_info}



# # keys are already lowercase breed names



# # -------------------------------------------
# # POSTGRESQL DATABASE CONNECTION
# # -------------------------------------------
# PG_HOST = "localhost"
# PG_PORT = "5432"
# PG_DB = "dog_app"
# PG_USER = "postgres"
# PG_PASSWORD = "root123"

# conn = psycopg2.connect(
#     host=PG_HOST,
#     port=PG_PORT,
#     database=PG_DB,
#     user=PG_USER,
#     password=PG_PASSWORD
# )
# conn.autocommit = True
# cursor = conn.cursor()

# # Create users table
# cursor.execute("""
# CREATE TABLE IF NOT EXISTS users (
#     id SERIAL PRIMARY KEY,
#     username VARCHAR(100) UNIQUE NOT NULL,
#     password TEXT NOT NULL
# );
# """)



# cursor.execute("""
# CREATE TABLE IF NOT EXISTS predictions (
#     id SERIAL PRIMARY KEY,
#     user_id INTEGER REFERENCES users(id),
#     breed VARCHAR(200),
#     confidence FLOAT,
#     timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# );
# """)

# # -------------------------------------------
# # JWT SECURITY CONFIG
# # -------------------------------------------
# JWT_SECRET = "CHANGE_THIS_SECRET_KEY"
# JWT_ALGO = "HS256"

# def generate_token(user_id):
#     payload = {
#         "user_id": user_id,
#         "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=6),
#         "iat": datetime.datetime.utcnow()
#     }
#     return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)

# def token_required(f):
#     @wraps(f)
#     def wrapper(*args, **kwargs):
#         token = request.headers.get("Authorization")

#         if not token:
#             return jsonify({"error": "Missing token"}), 401

#         try:
#             decoded = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
#             request.user_id = decoded["user_id"]
#         except jwt.ExpiredSignatureError:
#             return jsonify({"error": "Token expired"}), 401
#         except jwt.InvalidTokenError:
#             return jsonify({"error": "Invalid token"}), 401

#         return f(*args, **kwargs)
#     return wrapper


# # -------------------------------------------
# # AUTH ENDPOINTS
# # -------------------------------------------

# # REGISTER
# @app.route("/register", methods=["POST"])
# def register():
#     data = request.json
#     username = data.get("username")
#     password = data.get("password")

#     if not username or not password:
#         return jsonify({"error": "username and password required"}), 400

#     cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
#     if cursor.fetchone():
#         return jsonify({"error": "User already exists"}), 409

#     hashed_pw = generate_password_hash(password)
#     cursor.execute(
#         "INSERT INTO users (username, password) VALUES (%s, %s)",
#         (username, hashed_pw)
#     )

#     return jsonify({"message": "Registration successful"}), 201


# # LOGIN
# @app.route("/login", methods=["POST"])
# def login():
#     data = request.json
#     username = data.get("username")
#     password = data.get("password")

#     cursor.execute("SELECT id, password FROM users WHERE username=%s", (username,))
#     user = cursor.fetchone()

#     if not user:
#         return jsonify({"error": "Invalid username"}), 401

#     user_id, hashed_pw = user

#     if not check_password_hash(hashed_pw, password):
#         return jsonify({"error": "Incorrect password"}), 401

#     token = generate_token(user_id)

#     return jsonify({
#         "message": "Login successful",
#         "user_id": user_id,
#         "token": token
#     })


# # -------------------------------------------
# # LAZY MODEL LOAD
# # -------------------------------------------
# model = None

# def load_model():
#     global model
#     if model is None:
#         print("Loading model…")

#         m = timm.create_model(
#             "mobilenetv3_large_100",
#             pretrained=False,
#             num_classes=num_classes
#         )

#         ckpt = torch.load(MODEL_PATH, map_location=device)
#         state = ckpt.get("state", ckpt)
#         m.load_state_dict(state)

#         m.to(device)
#         m.eval()

#         model = m
#         print("Model loaded successfully!")


# preprocess = T.Compose([
#     T.Resize((224, 224)),
#     T.ToTensor()
# ])

# def preprocess_image(image):
#     image = image.convert("RGB")
#     tensor = preprocess(image)
#     return tensor.unsqueeze(0).to(device)


# def is_dog_image(image: Image.Image) -> bool:
#     """
#     Uses Gemini Vision to detect whether the image contains a dog.
#     Returns True if dog detected, else False.
#     """

#     prompt = """
#     Look at the image and answer ONLY with:
#     YES - if the image contains a dog
#     NO - if it does not contain a dog
#     """

#     model = genai.GenerativeModel("gemini-2.5-flash")

#     response = model.generate_content(
#         [prompt, image],
#         generation_config={"temperature": 0}
#     )

#     answer = response.text.strip().upper()

#     return answer == "YES"




# # ===========================================================
# # 1️⃣ PREDICT — PROTECTED ENDPOINT
# # ===========================================================
# @app.route("/predict", methods=["POST"])
# @token_required
# def predict():
#     load_model()

#     if "file" not in request.files:
#         return jsonify({"error": "No image uploaded"}), 400

#     try:
#         image = Image.open(request.files["file"])
#     except Exception as e:
#         return jsonify({"error": f"Invalid image: {e}"}), 400

#     tensor = preprocess_image(image)

#     with torch.no_grad():
#         outputs = model(tensor)
#         probs = torch.softmax(outputs, dim=1)[0]

#     # ensure top-5 sorted in descending order
#     # ensure top-5 sorted in descending order
#     top_vals, top_idx = torch.topk(probs, 5, largest=True, sorted=True)

#     top5 = []
#     for val, idx in zip(top_vals, top_idx):
#         b = class_names[idx.item()]
#         c = round(float(val.item() * 100), 2)
#         top5.append({"breed": b, "confidence": c})

#     # ✅ Save only the TOP-1 prediction
#     top_prediction = top5[0]

#     cursor.execute(
#         "INSERT INTO predictions (user_id, breed, confidence) VALUES (%s, %s, %s)",
#         (request.user_id, top_prediction["breed"], top_prediction["confidence"])
#     )

#     return jsonify({
#         "predicted": top_prediction,
#         "top5": top5
#     })

# # @app.route("/predict", methods=["POST"])
# # @token_required
# # def predict():
# #     load_model()

# #     if "file" not in request.files:
# #         return jsonify({"error": "No image uploaded"}), 400

# #     try:
# #         image = Image.open(request.files["file"])
# #     except Exception as e:
# #         return jsonify({"error": f"Invalid image: {e}"}), 400

# #     # ✅ STEP 1: Gemini validation
# #     try:
# #         is_dog = is_dog_image(image)
# #     except Exception as e:
# #         return jsonify({"error": f"Gemini vision error: {e}"}), 500

# #     if not is_dog:
# #         return jsonify({
# #             "error": "The uploaded image does not appear to be a dog. Please upload a dog image."
# #         }), 400

# #     # ✅ STEP 2: Only now run ML model
# #     tensor = preprocess_image(image)

# #     with torch.no_grad():
# #         outputs = model(tensor)
# #         probs = torch.softmax(outputs, dim=1)[0]

# #     top_vals, top_idx = torch.topk(probs, 5, largest=True, sorted=True)

# #     top5 = []
# #     for val, idx in zip(top_vals, top_idx):
# #         b = class_names[idx.item()]
# #         c = round(float(val.item() * 100), 2)
# #         top5.append({"breed": b, "confidence": c})

# #     top_prediction = top5[0]

# #     cursor.execute(
# #         "INSERT INTO predictions (user_id, breed, confidence) VALUES (%s, %s, %s)",
# #         (request.user_id, top_prediction["breed"], top_prediction["confidence"])
# #     )

# #     return jsonify({
# #         "predicted": top_prediction,
# #         "top5": top5
# #     })



# @app.route("/history", methods=["GET"])
# @token_required
# def history():
#     cursor.execute("SELECT breed, confidence, timestamp FROM predictions WHERE user_id=%s ORDER BY timestamp DESC", 
#                    (request.user_id,))
#     rows = cursor.fetchall()

#     history = [
#         {"breed": r[0], "confidence": r[1], "timestamp": r[2].isoformat()}
#         for r in rows
#     ]

#     return jsonify(history)

# # ===========================================================
# # 2️⃣ BREED INFO — PROTECTED
# # ===========================================================
# @app.route("/breed-info", methods=["GET"])
# @token_required
# def breed_info_api():
#     breed = request.args.get("breed", "").lower()
#     if not breed:
#         return jsonify({"error": "Missing breed parameter"}), 400
#     if breed not in breed_info:
#         return jsonify({"error": "Breed not found"}), 404
#     return jsonify(breed_info[breed])



# # ===========================================================
# # 4️⃣ DIET PLAN — PROTECTED
# # ===========================================================
# # @app.route("/diet-plan", methods=["GET"])
# # @token_required
# # def get_diet_plan():
# #     breed = request.args.get("breed", "")
# #     stage = request.args.get("stage", "")

# #     if not breed or not stage:
# #         return jsonify({"error": "breed and stage required"}), 400

# #     prompt = f"""
# #     You are a professional veterinary nutritionist.

# #     Generate a DAILY diet plan for a dog.

# #     Breed: {breed}
# #     Life Stage: {stage}

# #     Rules:
# #     - Use REAL food dishes (not generic terms)
# #     - Include:
# #         Breakfast
# #         Lunch
# #         Evening Snack
# #         Dinner
# #     - Mention approximate portion sizes
# #     - Avoid toxic foods for dogs
# #     - Suitable for Indian households
# #     - Keep it practical and safe

# #     Respond ONLY in JSON with keys:
# #     breakfast, lunch, snack, dinner, notes
# #     """

# #     model = genai.GenerativeModel("gemini-2.5-flash")

# #     response = model.generate_content(
# #         prompt,
# #         generation_config={"temperature": 0.4}
# #     )

# #     try:
# #         diet_json = json.loads(response.text)
# #     except Exception:
# #         return jsonify({"error": "Failed to generate diet plan"}), 500

# #     return jsonify(diet_json)
# @app.route("/diet-plan", methods=["GET"])
# @token_required
# def get_diet_plan():
#     stage = request.args.get("stage", "")

#     if not stage:
#         return jsonify({"error": "stage required"}), 400

#     with open("data/general_diet_plan.json") as f:
#         data = json.load(f)

#     if stage not in data:
#         return jsonify({"error": "Invalid stage"}), 400

#     plan = data[stage]

#     # 🔥 normalize keys
#     normalized = {k.lower(): v for k, v in plan.items()}

#     return jsonify(normalized)


# # ===========================================================
# # 3️⃣ CHATBOT — PROTECTED
# # ===========================================================
# def is_dog_related_query(user_input: str) -> bool:
#     """
#     Uses Gemini to dynamically decide whether a question is dog-related.
#     Returns True only if the question is about dogs.
#     """

#     prompt = f"""
#     You are a strict classifier.

#     Decide whether the following question is related to:
#     - dogs
#     - dog breeds
#     - dog food, diet, health, training, behavior, grooming

#     Question:
#     "{user_input}"

#     Respond ONLY with:
#     YES
#     or
#     NO
#     """

#     model = genai.GenerativeModel("gemini-2.5-flash")

#     response = model.generate_content(
#         prompt,
#         generation_config={"temperature": 0}
#     )

#     answer = response.text.strip().upper()
#     return answer == "YES"

# genai.configure(api_key="AIzaSyBoWgxWTFCF8r7U75XE0jeNgcyxyCQQTXc")


# # @app.route("/chat", methods=["POST"])
# # @token_required
# # def chat():
# #     breed = request.form.get("breed", "").lower()
# #     user_input = request.form.get("user_input", "")

# #     if not breed or not user_input:
# #         return jsonify({"error": "Missing breed or user_input"}), 400

# #     if breed in breed_info:
# #         info = breed_info[breed]
# #         for key, value in info.items():
# #             if key.lower() in user_input.lower():
# #                 return jsonify({"reply": f"{key}: {value}"})
# #         if any(term in user_input.lower() for term in ["info", "details", "about"]):
# #             return jsonify({"reply": json.dumps(info, indent=2)})

# #     prompt = f"""
# #     The user is asking about {breed}.
# #     Question: {user_input}
# #     Provide an accurate response.
# #     """

# #     gem = genai.GenerativeModel("gemini-2.5-flash")
# #     reply = gem.generate_content(prompt).text

# #     return jsonify({"reply": reply})



# @app.route("/chat", methods=["POST"])
# @token_required
# def chat():
#     breed = request.form.get("breed")
#     breed = breed.lower() if breed else "general dog"

#     user_input = request.form.get("user_input", "")

#     if not user_input:
#         return jsonify({"error": "Missing user_input"}), 400

#     # -------------------------------------------------
#     # 🧠 DYNAMIC DOG-INTENT CHECK (LLM-BASED)
#     # -------------------------------------------------
#     try:
#         is_dog_question = is_dog_related_query(user_input)
#     except Exception as e:
#         return jsonify({"error": f"Intent detection failed: {e}"}), 500

#     if not is_dog_question:
#         return jsonify({
#             "reply": "❌ I can only answer questions related to dogs and their care."
#         }), 403

#     # -------------------------------------------------
#     # ⚡ RULE-BASED RESPONSE (FAST PATH)
#     # -------------------------------------------------
#     if breed in breed_info:
#         info = breed_info[breed]
#         for key, value in info.items():
#             if key.lower() in user_input.lower():
#                 return jsonify({"reply": f"{key}: {value}"})

#         if any(term in user_input.lower() for term in ["info", "details", "about"]):
#             return jsonify({"reply": json.dumps(info, indent=2)})

#     # -------------------------------------------------
#     # 🤖 GEMINI — DOG QUESTIONS ONLY
#     # -------------------------------------------------
#     prompt = f"""
#     You are a veterinary and dog-breed expert.
#     Answer ONLY dog-related questions.

#     Breed: {breed if breed else "general dog"}
#     Question: {user_input}
#     """

#     model = genai.GenerativeModel("gemini-2.5-flash")
#     reply = model.generate_content(prompt).text

#     return jsonify({"reply": reply})






# @app.route("/map-redirect", methods=["GET"])
# @token_required
# def map_redirect():
#     location = request.args.get("location")
#     service = request.args.get("service")

#     if not location or not service:
#         return jsonify({"error": "location and service required"}), 400

#     url = f"https://www.google.com/maps/search/{service}+near+{location}"
#     return jsonify({"url": url})


# # -------------------------------------------
# # RUN SERVER
# # -------------------------------------------
# if __name__ == "__main__":
#     print("Server running at http://127.0.0.1:5000")
#     app.run(host="127.0.0.1", port=5000, debug=True)


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

genai.configure(api_key="AIzaSyBoWgxWTFCF8r7U75XE0jeNgcyxyCQQTXc")

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
    password="root123"
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

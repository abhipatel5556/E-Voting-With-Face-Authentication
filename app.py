import base64
import hashlib
import json
import os
import random
import smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from functools import wraps
from typing import Optional

import cv2
import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

try:
    import face_recognition
    FACE_LIB_AVAILABLE = True
except Exception:
    face_recognition = None
    FACE_LIB_AVAILABLE = False


app = Flask(__name__, template_folder="templates", static_folder="templates/static", static_url_path="/static")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-session-secret")

# SQLite default with MySQL optional fallback
# Example MySQL: mysql+pymysql://user:pass@localhost:3306/evoting
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///evoting.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

OTP_TTL_MINUTES = int(os.getenv("OTP_TTL_MINUTES", "5"))
OTP_DEV_MODE = os.getenv("OTP_DEV_MODE", "false").lower() == "true"

CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

bcrypt = Bcrypt(app)
db = SQLAlchemy(app)


class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    username = db.Column(db.String(120), unique=True, index=True)
    unique_id = db.Column(db.String(120), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255))
    facial_image_data = db.Column(db.Text, nullable=False)
    registration_status = db.Column(db.String(20), default="pending", nullable=False)
    role = db.Column(db.String(20), default="voter", nullable=False)
    otp_code = db.Column(db.String(255))  # sha256(otp)|expiry_iso
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)


class Election(db.Model):
    __tablename__ = "elections"
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime, nullable=False)
    status = db.Column(db.String(30), default="scheduled", nullable=False)
    created_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)


class Candidate(db.Model):
    __tablename__ = "candidates"
    id = db.Column(db.Integer, primary_key=True)
    election_id = db.Column(db.Integer, db.ForeignKey("elections.id"), nullable=False)
    name = db.Column(db.String(120), nullable=False)
    photo_url = db.Column(db.String(500))
    details = db.Column(db.Text)


class VoterEligibility(db.Model):
    __tablename__ = "voter_eligibility"
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), primary_key=True)
    election_id = db.Column(db.Integer, db.ForeignKey("elections.id"), primary_key=True)
    has_voted = db.Column(db.Boolean, default=False, nullable=False)
    face_verified = db.Column(db.Boolean, default=False, nullable=False)


class Vote(db.Model):
    __tablename__ = "votes"
    id = db.Column(db.Integer, primary_key=True)
    election_id = db.Column(db.Integer, db.ForeignKey("elections.id"), nullable=False)
    candidate_id = db.Column(db.Integer, db.ForeignKey("candidates.id"), nullable=False)
    voter_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    cast_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)


class Block(db.Model):
    __tablename__ = "vote_blockchain"
    id = db.Column(db.Integer, primary_key=True)
    previous_hash = db.Column(db.String(128), nullable=False)
    payload = db.Column(db.Text, nullable=False)
    block_hash = db.Column(db.String(128), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)


class AuditTrail(db.Model):
    __tablename__ = "audit_trail"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    action = db.Column(db.String(120), nullable=False)
    details = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)


def log_action(action: str, details: str, user_id: Optional[int] = None) -> None:
    db.session.add(AuditTrail(user_id=user_id, action=action, details=details))


def db_commit():
    db.session.commit()


def db_get_user_by_email(email: str) -> Optional[User]:
    return User.query.filter_by(email=email).first()


def db_get_user_by_username_or_unique(identifier: str) -> Optional[User]:
    return User.query.filter((User.username == identifier) | (User.unique_id == identifier)).first()


def decode_base64_image(img_b64: str) -> Optional[np.ndarray]:
    try:
        raw = img_b64.split(",", 1)[1] if "," in img_b64 else img_b64
        image_bytes = base64.b64decode(raw)
        np_array = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def encode_image_to_base64(img: np.ndarray) -> str:
    ok, buff = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Unable to encode image")
    return base64.b64encode(buff.tobytes()).decode("utf-8")


def extract_primary_face(gray_image: np.ndarray) -> Optional[np.ndarray]:
    faces = CASCADE.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(90, 90))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = gray_image[y : y + h, x : x + w]
    face = cv2.resize(face, (160, 160))
    return cv2.equalizeHist(face)


def compare_faces(reference_b64: str, probe_b64: str) -> bool:
    # Required integration with face_recognition library with fallback to OpenCV similarity.
    reference_img = decode_base64_image(reference_b64)
    probe_img = decode_base64_image(probe_b64)
    if reference_img is None or probe_img is None:
        return False

    if FACE_LIB_AVAILABLE:
        ref_rgb = cv2.cvtColor(reference_img, cv2.COLOR_GRAY2RGB)
        probe_rgb = cv2.cvtColor(probe_img, cv2.COLOR_GRAY2RGB)
        ref_locs = face_recognition.face_locations(ref_rgb)
        probe_locs = face_recognition.face_locations(probe_rgb)
        if len(ref_locs) > 0 and len(probe_locs) > 0:
            ref_enc = face_recognition.face_encodings(ref_rgb, known_face_locations=[ref_locs[0]])
            probe_enc = face_recognition.face_encodings(probe_rgb, known_face_locations=[probe_locs[0]])
            if ref_enc and probe_enc:
                distance = np.linalg.norm(ref_enc[0] - probe_enc[0])
                return distance < 0.55

    # fallback path
    ref_face = extract_primary_face(reference_img)
    if ref_face is None:
        ref_face = cv2.equalizeHist(cv2.resize(reference_img, (160, 160)))
    probe_face = extract_primary_face(probe_img)
    if probe_face is None:
        return False

    # robust histogram cosine similarity fallback
    ref_hist = cv2.calcHist([ref_face], [0], None, [64], [0, 256]).flatten()
    probe_hist = cv2.calcHist([probe_face], [0], None, [64], [0, 256]).flatten()
    ref_hist = ref_hist / (np.linalg.norm(ref_hist) + 1e-9)
    probe_hist = probe_hist / (np.linalg.norm(probe_hist) + 1e-9)
    similarity = float(np.dot(ref_hist, probe_hist))
    return similarity >= 0.90


def hash_otp(otp: str) -> str:
    return hashlib.sha256(otp.encode()).hexdigest()


def set_user_otp(user: User, otp: str) -> None:
    expiry = datetime.now(timezone.utc) + timedelta(minutes=OTP_TTL_MINUTES)
    user.otp_code = f"{hash_otp(otp)}|{expiry.isoformat()}"


def verify_user_otp(user: User, otp: str) -> bool:
    if not user.otp_code or "|" not in user.otp_code:
        return False
    hashed, expiry_iso = user.otp_code.split("|", 1)
    try:
        expiry = datetime.fromisoformat(expiry_iso)
    except ValueError:
        return False
    if expiry < datetime.now(timezone.utc):
        return False
    return hashed == hash_otp(otp)


def send_otp_email(email: str, otp: str) -> bool:
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("SMTP_SENDER", username or "no-reply@evoting.local")

    if not host or not username or not password:
        return False

    message = EmailMessage()
    message["Subject"] = "Your E-Voting OTP Code"
    message["From"] = sender
    message["To"] = email
    message.set_content(f"Your OTP code is {otp}. It expires in {OTP_TTL_MINUTES} minutes.")

    with smtplib.SMTP(host, port, timeout=15) as server:
        server.starttls()
        server.login(username, password)
        server.send_message(message)
    return True


def append_vote_block(vote: Vote) -> str:
    last_block = Block.query.order_by(Block.id.desc()).first()
    previous_hash = last_block.block_hash if last_block else "GENESIS"
    payload = json.dumps(
        {
            "vote_id": vote.id,
            "election_id": vote.election_id,
            "candidate_id": vote.candidate_id,
            "voter_id": vote.voter_id,
            "cast_at": vote.cast_at.isoformat(),
        },
        sort_keys=True,
    )
    block_hash = hashlib.sha256(f"{previous_hash}|{payload}".encode()).hexdigest()
    db.session.add(Block(previous_hash=previous_hash, payload=payload, block_hash=block_hash))
    return block_hash


def blockchain_verify() -> dict:
    blocks = Block.query.order_by(Block.id.asc()).all()
    previous = "GENESIS"
    for blk in blocks:
        recomputed = hashlib.sha256(f"{previous}|{blk.payload}".encode()).hexdigest()
        if recomputed != blk.block_hash:
            return {"valid": False, "failed_block_id": blk.id}
        previous = blk.block_hash
    return {"valid": True, "length": len(blocks)}


def login_required(role: Optional[str] = None):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            user_id = session.get("user_id")
            if not user_id:
                return jsonify({"error": "Authentication required"}), 401
            user = User.query.get(user_id)
            if not user:
                return jsonify({"error": "Invalid session"}), 401
            if role and user.role != role:
                return jsonify({"error": "Forbidden"}), 403
            request.current_user = user
            return fn(*args, **kwargs)

        return wrapper

    return decorator


@app.errorhandler(SQLAlchemyError)
def handle_db_error(error):
    db.session.rollback()
    return jsonify({"error": "Database error", "details": str(error)}), 500


# ---------- Front-end pages ----------
@app.get("/")
def index_page():
    return render_template("index.html")


@app.get("/login")
def login_page_alias():
    return render_template("index.html")


@app.get("/register")
def register_page_alias():
    return render_template("registration_page.html")


@app.get("/<path:page_name>")
def static_pages(page_name: str):
    if not page_name.endswith(".html"):
        return jsonify({"error": "Not found"}), 404
    template_path = os.path.join(app.template_folder, page_name)
    if not os.path.exists(template_path):
        return jsonify({"error": "Page not found"}), 404
    return render_template(page_name)


# ---------- Auth + Session APIs ----------
@app.post("/api/auth/register")
def register_api():
    payload = request.get_json(force=True)
    required = ["name", "unique_id", "email", "password", "facial_image_data"]
    missing = [field for field in required if not payload.get(field)]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    username = payload.get("username") or payload["unique_id"]
    if User.query.filter(
        (User.unique_id == payload["unique_id"]) | (User.email == payload["email"]) | (User.username == username)
    ).first():
        return jsonify({"error": "User with email/unique ID/username already exists"}), 409

    reg_image = decode_base64_image(payload["facial_image_data"])
    if reg_image is None:
        return jsonify({"error": "Invalid facial_image_data format"}), 400
    normalized_face = extract_primary_face(reg_image)
    if normalized_face is None:
        return jsonify({"error": "No face detected. Please capture a clear frontal face image."}), 400

    user = User(
        name=payload["name"],
        username=username,
        unique_id=payload["unique_id"],
        email=payload["email"],
        password_hash=bcrypt.generate_password_hash(payload["password"]).decode("utf-8"),
        facial_image_data=encode_image_to_base64(normalized_face),
        registration_status="pending",
        role=payload.get("role", "voter"),
    )
    db.session.add(user)
    db_commit()
    log_action("register", "User registration submitted", user.id)
    db_commit()
    return jsonify({"message": "Registered successfully. Awaiting admin approval.", "user_id": user.id}), 201


@app.post("/api/auth/login-password")
def login_password():
    payload = request.get_json(force=True)
    identifier = payload.get("username") or payload.get("unique_id")
    password = payload.get("password")
    if not identifier or not password:
        return jsonify({"error": "username/unique_id and password are required"}), 400

    user = db_get_user_by_username_or_unique(identifier)
    if not user or not user.password_hash or not bcrypt.check_password_hash(user.password_hash, password):
        return jsonify({"error": "Invalid credentials"}), 401
    if user.registration_status != "approved":
        return jsonify({"error": "User not approved yet"}), 403

    session["user_id"] = user.id
    session["role"] = user.role
    log_action("login_password", "Password login success", user.id)
    db_commit()

    next_page = "admin_dashboard.html" if user.role == "admin" else "voter_dashboard.html"
    return jsonify({"message": "Login successful", "role": user.role, "redirect": f"/{next_page}"})




@app.post("/api/auth/login-face")
def login_face():
    payload = request.get_json(force=True)
    unique_id = payload.get("unique_id") or payload.get("username")
    live_facial_image_data = payload.get("live_facial_image_data")
    if not unique_id or not live_facial_image_data:
        return jsonify({"error": "unique_id/username and live_facial_image_data are required"}), 400

    user = db_get_user_by_username_or_unique(unique_id)
    if not user or user.registration_status != "approved":
        return jsonify({"error": "Authentication failed or user not approved"}), 401
    if not compare_faces(user.facial_image_data, live_facial_image_data):
        return jsonify({"error": "Face verification failed"}), 401

    session["user_id"] = user.id
    session["role"] = user.role
    log_action("login_face", "Face login success", user.id)
    db_commit()
    next_page = "admin_dashboard.html" if user.role == "admin" else "voter_dashboard.html"
    return jsonify({"message": "Login successful", "role": user.role, "redirect": f"/{next_page}"})


@app.post("/api/auth/logout")
def logout():
    session.clear()
    return jsonify({"message": "Logged out"})


@app.post("/api/auth/login-otp-request")
def otp_request():
    payload = request.get_json(force=True)
    email = payload.get("email")
    if not email:
        return jsonify({"error": "email is required"}), 400

    user = db_get_user_by_email(email)
    if not user or user.registration_status != "approved":
        return jsonify({"error": "No approved user found with this email"}), 404

    otp = f"{random.randint(100000, 999999)}"
    set_user_otp(user, otp)

    sent = False
    try:
        sent = send_otp_email(email, otp)
    except Exception as err:
        log_action("otp_send_error", f"OTP email failed: {err}", user.id)

    db_commit()
    if sent:
        return jsonify({"message": "OTP sent to your email address."})
    if OTP_DEV_MODE:
        return jsonify({"message": "OTP generated (dev mode).", "otp_code": otp, "warning": "SMTP is not configured."})
    return jsonify({"error": "Unable to send OTP email. Configure SMTP settings."}), 503


@app.post("/api/auth/login-otp-verify")
def otp_verify():
    payload = request.get_json(force=True)
    email = payload.get("email")
    otp_code = payload.get("otp_code")
    user = db_get_user_by_email(email) if email else None
    if not user or user.registration_status != "approved":
        return jsonify({"error": "Invalid email or account not approved"}), 404
    if not otp_code or not verify_user_otp(user, otp_code):
        return jsonify({"error": "Invalid or expired OTP"}), 400

    user.otp_code = None
    session["user_id"] = user.id
    session["role"] = user.role
    log_action("login_otp", "OTP login success", user.id)
    db_commit()

    next_page = "admin_dashboard.html" if user.role == "admin" else "voter_dashboard.html"
    return jsonify({"message": "Login successful", "role": user.role, "redirect": f"/{next_page}"})


@app.post("/face_verify")
@login_required(role="voter")
def face_verify():
    payload = request.get_json(force=True)
    election_id = payload.get("election_id")
    live_facial_image_data = payload.get("live_facial_image_data")
    if not election_id or not live_facial_image_data:
        return jsonify({"error": "election_id and live_facial_image_data are required"}), 400

    user = request.current_user
    eligibility = VoterEligibility.query.filter_by(user_id=user.id, election_id=election_id).first()
    if not eligibility:
        return jsonify({"error": "Not eligible for this election"}), 403

    if not compare_faces(user.facial_image_data, live_facial_image_data):
        return jsonify({"error": "Face verification failed"}), 401

    eligibility.face_verified = True
    db_commit()
    return jsonify({"message": "Face verified successfully"})


# ---------- Voter APIs ----------
@app.get("/api/profile")
@login_required()
def profile():
    user = request.current_user
    return jsonify(
        {
            "user_id": user.id,
            "name": user.name,
            "username": user.username,
            "unique_id": user.unique_id,
            "email": user.email,
            "registration_status": user.registration_status,
            "role": user.role,
        }
    )


@app.get("/api/voter/elections")
@login_required(role="voter")
def voter_elections():
    user = request.current_user
    rows = (
        db.session.query(Election, VoterEligibility)
        .join(VoterEligibility, VoterEligibility.election_id == Election.id)
        .filter(VoterEligibility.user_id == user.id)
        .all()
    )
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    result = []
    for election, eligibility in rows:
        status = election.status
        if election.start_time <= now <= election.end_time:
            status = "ongoing"
        elif now > election.end_time:
            status = "completed"
        result.append(
            {
                "election_id": election.id,
                "title": election.title,
                "description": election.description,
                "start_time": election.start_time.isoformat(),
                "end_time": election.end_time.isoformat(),
                "status": status,
                "has_voted": eligibility.has_voted,
                "face_verified": eligibility.face_verified,
            }
        )
    return jsonify(result)


@app.post("/api/voter/elections/<int:election_id>/vote")
@login_required(role="voter")
def cast_vote(election_id: int):
    user = request.current_user
    payload = request.get_json(force=True)
    candidate_id = payload.get("candidate_id")

    eligibility = VoterEligibility.query.filter_by(user_id=user.id, election_id=election_id).first()
    if not eligibility:
        return jsonify({"error": "Not eligible for this election"}), 403
    if eligibility.has_voted:
        return jsonify({"error": "Vote already cast"}), 409
    if not eligibility.face_verified:
        return jsonify({"error": "Face verification required before voting"}), 403

    candidate = Candidate.query.filter_by(id=candidate_id, election_id=election_id).first()
    if not candidate:
        return jsonify({"error": "Invalid candidate"}), 400

    vote = Vote(election_id=election_id, candidate_id=candidate.id, voter_id=user.id)
    db.session.add(vote)
    db.session.flush()
    eligibility.has_voted = True
    block_hash = append_vote_block(vote)
    log_action("vote_cast", f"Vote {vote.id} recorded with block {block_hash}", user.id)
    db_commit()
    return jsonify({"message": "Vote cast successfully", "vote_id": vote.id, "block_hash": block_hash}), 201


@app.get("/api/public/results/<int:election_id>")
def public_results(election_id: int):
    election = Election.query.get_or_404(election_id)
    rows = (
        db.session.query(Candidate.id, Candidate.name, db.func.count(Vote.id).label("votes"))
        .outerjoin(Vote, Vote.candidate_id == Candidate.id)
        .filter(Candidate.election_id == election_id)
        .group_by(Candidate.id)
        .all()
    )
    return jsonify(
        {
            "election_id": election.id,
            "title": election.title,
            "results": [{"candidate_id": r.id, "name": r.name, "vote_count": int(r.votes)} for r in rows],
        }
    )


# ---------- Admin APIs ----------
@app.get("/api/admin/pending-registrations")
@login_required(role="admin")
def pending_registrations():
    users = User.query.filter_by(registration_status="pending").all()
    return jsonify(
        [
            {
                "user_id": u.id,
                "name": u.name,
                "username": u.username,
                "unique_id": u.unique_id,
                "email": u.email,
                "created_at": u.created_at.isoformat(),
            }
            for u in users
        ]
    )


@app.put("/api/admin/users/<int:user_id>/registration-status")
@login_required(role="admin")
def registration_status(user_id: int):
    payload = request.get_json(force=True)
    status_value = payload.get("registration_status")
    if status_value not in {"approved", "rejected", "pending"}:
        return jsonify({"error": "registration_status must be approved, rejected, or pending"}), 400
    user = User.query.get_or_404(user_id)
    user.registration_status = status_value
    log_action("registration_status_updated", f"Set user {user.id} to {status_value}", request.current_user.id)
    db_commit()
    return jsonify({"message": "Registration status updated", "user_id": user.id, "registration_status": user.registration_status})


@app.post("/api/admin/elections")
@login_required(role="admin")
def create_election():
    payload = request.get_json(force=True)
    try:
        election = Election(
            title=payload["title"],
            description=payload.get("description"),
            start_time=datetime.fromisoformat(payload["start_time"]),
            end_time=datetime.fromisoformat(payload["end_time"]),
            status=payload.get("status", "scheduled"),
            created_by=request.current_user.id,
        )
    except (KeyError, ValueError):
        return jsonify({"error": "Invalid payload. Required: title, start_time, end_time (ISO format)."}), 400

    db.session.add(election)
    db_commit()
    return jsonify({"election_id": election.id, "message": "Election created"}), 201


@app.get("/api/admin/elections")
@login_required(role="admin")
def admin_elections():
    elections = Election.query.order_by(Election.start_time.desc()).all()
    return jsonify(
        [
            {
                "election_id": e.id,
                "title": e.title,
                "description": e.description,
                "status": e.status,
                "start_time": e.start_time.isoformat(),
                "end_time": e.end_time.isoformat(),
            }
            for e in elections
        ]
    )


@app.post("/api/admin/elections/<int:election_id>/candidates")
@login_required(role="admin")
def add_candidate(election_id: int):
    payload = request.get_json(force=True)
    election = Election.query.get_or_404(election_id)
    if not payload.get("name"):
        return jsonify({"error": "Candidate name is required"}), 400

    candidate = Candidate(
        election_id=election.id,
        name=payload["name"],
        photo_url=payload.get("photo_url"),
        details=payload.get("details"),
    )
    db.session.add(candidate)
    db_commit()
    return jsonify({"candidate_id": candidate.id, "message": "Candidate added"}), 201


@app.post("/api/admin/elections/<int:election_id>/eligibility")
@login_required(role="admin")
def set_eligibility(election_id: int):
    payload = request.get_json(force=True)
    user_ids = payload.get("user_ids", [])
    if not isinstance(user_ids, list):
        return jsonify({"error": "user_ids must be an array"}), 400

    Election.query.get_or_404(election_id)
    VoterEligibility.query.filter_by(election_id=election_id).delete()

    created = 0
    for user_id in user_ids:
        user = User.query.get(user_id)
        if user and user.registration_status == "approved":
            db.session.add(VoterEligibility(user_id=user.id, election_id=election_id, has_voted=False, face_verified=False))
            created += 1

    db_commit()
    return jsonify({"message": "Eligibility updated", "eligibility_count": created})


@app.get("/api/admin/elections/<int:election_id>/monitor")
@login_required(role="admin")
def monitor_election(election_id: int):
    total_eligible = VoterEligibility.query.filter_by(election_id=election_id).count()
    votes_cast = Vote.query.filter_by(election_id=election_id).count()
    turnout = round((votes_cast / total_eligible) * 100, 2) if total_eligible else 0
    return jsonify(
        {
            "election_id": election_id,
            "total_eligible_voters": total_eligible,
            "votes_cast": votes_cast,
            "turnout_percentage": turnout,
        }
    )


@app.get("/api/admin/audittrail")
@login_required(role="admin")
def audit_trail():
    logs = AuditTrail.query.order_by(AuditTrail.created_at.desc()).limit(500).all()
    return jsonify(
        [
            {
                "id": log.id,
                "user_id": log.user_id,
                "action": log.action,
                "details": log.details,
                "created_at": log.created_at.isoformat(),
            }
            for log in logs
        ]
    )


@app.get("/api/admin/blockchain/verify")
@login_required(role="admin")
def verify_blockchain():
    return jsonify(blockchain_verify())


# Lightweight form wiring with existing UI (no redesign)
@app.post("/form/login")
def login_form_submit():
    data = request.form.to_dict() or request.get_json(silent=True) or {}
    with app.test_request_context(json={"username": data.get("username"), "password": data.get("password")}):
        pass
    return redirect(url_for("index_page"))


with app.app_context():
    db.create_all()
    # schema self-healing for old db files
    try:
        db.session.execute(text("ALTER TABLE users ADD COLUMN username VARCHAR(120)"))
        db.session.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_username ON users (username)"))
        db.session.execute(text("ALTER TABLE users ADD COLUMN password_hash VARCHAR(255)"))
        db.session.commit()
    except Exception:
        db.session.rollback()

    try:
        db.session.execute(text("ALTER TABLE voter_eligibility ADD COLUMN face_verified BOOLEAN DEFAULT 0"))
        db.session.commit()
    except Exception:
        db.session.rollback()

    admin_unique_id = os.getenv("DEFAULT_ADMIN_ID", "admin")
    admin_email = os.getenv("DEFAULT_ADMIN_EMAIL", "admin@example.com")
    admin_password = os.getenv("DEFAULT_ADMIN_PASSWORD", "admin123")
    admin = User.query.filter_by(unique_id=admin_unique_id).first()
    if not admin:
        blank = np.zeros((160, 160), dtype=np.uint8)
        cv2.circle(blank, (80, 70), 35, 180, 2)
        cv2.circle(blank, (66, 65), 3, 255, -1)
        cv2.circle(blank, (94, 65), 3, 255, -1)
        cv2.ellipse(blank, (80, 85), (18, 10), 0, 0, 180, 200, 2)
        db.session.add(
            User(
                name="System Admin",
                username="admin",
                unique_id=admin_unique_id,
                email=admin_email,
                password_hash=bcrypt.generate_password_hash(admin_password).decode("utf-8"),
                facial_image_data=encode_image_to_base64(blank),
                registration_status="approved",
                role="admin",
            )
        )
        db.session.commit()
    else:
        changed = False
        if not admin.username:
            admin.username = "admin"
            changed = True
        if not admin.password_hash:
            admin.password_hash = bcrypt.generate_password_hash(admin_password).decode("utf-8")
            changed = True
        if changed:
            db.session.commit()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)

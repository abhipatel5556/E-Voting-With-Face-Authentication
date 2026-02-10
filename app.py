import base64
import hashlib
import json
import os
import random
import smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Optional

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_jwt_extended import JWTManager, create_access_token, get_jwt_identity, jwt_required
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import SQLAlchemyError


app = Flask(__name__, template_folder="templates", static_folder="templates/static", static_url_path="/static")

# MySQL example: mysql+pymysql://user:password@localhost:3306/evoting
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///evoting.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "change-this-secret-in-production")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=2)

OTP_TTL_MINUTES = int(os.getenv("OTP_TTL_MINUTES", "5"))
OTP_DEV_MODE = os.getenv("OTP_DEV_MODE", "false").lower() == "true"

CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


db = SQLAlchemy(app)
jwt = JWTManager(app)


class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    unique_id = db.Column(db.String(120), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    facial_image_data = db.Column(db.Text, nullable=False)  # Stores normalized face crop as base64 PNG
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


def face_descriptor(face: np.ndarray) -> np.ndarray:
    # Histogram descriptor from 4x4 cells gives a practical, deterministic biometric signature.
    feats = []
    step = 40
    for y in range(0, 160, step):
        for x in range(0, 160, step):
            cell = face[y : y + step, x : x + step]
            hist = cv2.calcHist([cell], [0], None, [16], [0, 256]).flatten()
            hist = hist / (np.linalg.norm(hist) + 1e-9)
            feats.append(hist)
    return np.concatenate(feats)


def compare_faces(reference_b64: str, probe_b64: str) -> bool:
    reference_img = decode_base64_image(reference_b64)
    probe_img = decode_base64_image(probe_b64)
    if reference_img is None or probe_img is None:
        return False

    ref_face = extract_primary_face(reference_img)
    if ref_face is None:
        # If reference is already a cropped face template from registration, normalize directly.
        ref_face = cv2.equalizeHist(cv2.resize(reference_img, (160, 160)))

    probe_face = extract_primary_face(probe_img)
    if probe_face is None:
        return False

    ref_desc = face_descriptor(ref_face)
    probe_desc = face_descriptor(probe_face)
    similarity = float(np.dot(ref_desc, probe_desc) / ((np.linalg.norm(ref_desc) * np.linalg.norm(probe_desc)) + 1e-9))
    return similarity >= 0.84


def hash_otp(otp: str) -> str:
    return hashlib.sha256(otp.encode()).hexdigest()


def generate_otp() -> str:
    return f"{random.randint(100000, 999999)}"


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
    payload_dict = {
        "vote_id": vote.id,
        "election_id": vote.election_id,
        "candidate_id": vote.candidate_id,
        "voter_id": vote.voter_id,
        "cast_at": vote.cast_at.isoformat(),
    }
    payload = json.dumps(payload_dict, sort_keys=True)
    block_hash = hashlib.sha256(f"{previous_hash}|{payload}".encode()).hexdigest()
    db.session.add(Block(previous_hash=previous_hash, payload=payload, block_hash=block_hash))
    return block_hash


def get_current_user() -> User:
    identity = get_jwt_identity()
    user = User.query.filter_by(unique_id=identity).first()
    if not user:
        raise ValueError("Invalid user")
    return user


def admin_required() -> User:
    user = get_current_user()
    if user.role != "admin":
        raise PermissionError("Admin access required")
    return user


@app.errorhandler(SQLAlchemyError)
def handle_db_error(error):
    db.session.rollback()
    return jsonify({"error": "Database error", "details": str(error)}), 500


# ---------- Front-end pages ----------
@app.get("/")
def index_page():
    return render_template("index.html")


@app.get("/<path:page_name>")
def static_pages(page_name: str):
    if not page_name.endswith(".html"):
        return jsonify({"error": "Not found"}), 404
    template_path = os.path.join(app.template_folder, page_name)
    if not os.path.exists(template_path):
        return jsonify({"error": "Page not found"}), 404
    return render_template(page_name)


# ---------- Auth APIs ----------
@app.post("/api/auth/register")
def register():
    payload = request.get_json(force=True)
    required = ["name", "unique_id", "email", "facial_image_data"]
    missing = [field for field in required if not payload.get(field)]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    if User.query.filter((User.unique_id == payload["unique_id"]) | (User.email == payload["email"])).first():
        return jsonify({"error": "User with email or unique ID already exists"}), 409

    reg_image = decode_base64_image(payload["facial_image_data"])
    if reg_image is None:
        return jsonify({"error": "Invalid facial_image_data format"}), 400

    normalized_face = extract_primary_face(reg_image)
    if normalized_face is None:
        return jsonify({"error": "No face detected. Please capture a clear frontal face image."}), 400

    user = User(
        name=payload["name"],
        unique_id=payload["unique_id"],
        email=payload["email"],
        facial_image_data=encode_image_to_base64(normalized_face),
        registration_status="pending",
        role="voter",
    )
    db.session.add(user)
    log_action("register", "User registration submitted with biometric profile", user.id)
    db.session.commit()
    return jsonify({"message": "Registered successfully. Awaiting admin approval.", "user_id": user.id}), 201


@app.post("/api/auth/login-face")
def login_face():
    payload = request.get_json(force=True)
    unique_id = payload.get("unique_id")
    live_facial_image_data = payload.get("live_facial_image_data")
    if not unique_id or not live_facial_image_data:
        return jsonify({"error": "unique_id and live_facial_image_data are required"}), 400

    user = User.query.filter_by(unique_id=unique_id).first()
    if not user or user.registration_status != "approved":
        return jsonify({"error": "Authentication failed or user not approved"}), 401

    if not compare_faces(user.facial_image_data, live_facial_image_data):
        log_action("login_face_failed", "Biometric verification failed", user.id)
        db.session.commit()
        return jsonify({"error": "Face verification failed"}), 401

    token = create_access_token(identity=user.unique_id, additional_claims={"role": user.role})
    log_action("login_face", "Successful face login", user.id)
    db.session.commit()
    return jsonify({"access_token": token, "token_type": "bearer", "user_role": user.role})


@app.post("/api/auth/login-otp-request")
def otp_request():
    payload = request.get_json(force=True)
    email = payload.get("email")
    if not email:
        return jsonify({"error": "email is required"}), 400

    user = User.query.filter_by(email=email).first()
    if not user or user.registration_status != "approved":
        return jsonify({"error": "No approved user found with this email"}), 404

    otp = generate_otp()
    set_user_otp(user, otp)

    sent = False
    try:
        sent = send_otp_email(email, otp)
    except Exception as err:
        log_action("otp_send_error", f"OTP email failed: {err}", user.id)

    log_action("otp_requested", f"OTP generated for {email}", user.id)
    db.session.commit()

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
    user = User.query.filter_by(email=email).first()
    if not user or user.registration_status != "approved":
        return jsonify({"error": "Invalid email or account not approved"}), 404
    if not otp_code or not verify_user_otp(user, otp_code):
        return jsonify({"error": "Invalid or expired OTP"}), 400

    user.otp_code = None
    token = create_access_token(identity=user.unique_id, additional_claims={"role": user.role})
    log_action("login_otp", "Successful OTP login", user.id)
    db.session.commit()
    return jsonify({"access_token": token, "token_type": "bearer", "user_role": user.role})


@app.get("/api/profile")
@jwt_required()
def profile():
    user = get_current_user()
    return jsonify(
        {
            "user_id": user.id,
            "name": user.name,
            "unique_id": user.unique_id,
            "email": user.email,
            "registration_status": user.registration_status,
            "role": user.role,
        }
    )


@app.get("/api/voter/elections")
@jwt_required()
def voter_elections():
    user = get_current_user()
    rows = (
        db.session.query(Election, VoterEligibility)
        .join(VoterEligibility, VoterEligibility.election_id == Election.id)
        .filter(VoterEligibility.user_id == user.id)
        .all()
    )
    return jsonify(
        [
            {
                "election_id": election.id,
                "title": election.title,
                "description": election.description,
                "start_time": election.start_time.isoformat(),
                "end_time": election.end_time.isoformat(),
                "status": election.status,
                "has_voted": eligibility.has_voted,
            }
            for election, eligibility in rows
        ]
    )


@app.get("/api/voter/elections/<int:election_id>/ballot")
@jwt_required()
def ballot(election_id: int):
    election = Election.query.get_or_404(election_id)
    candidates = Candidate.query.filter_by(election_id=election.id).all()
    return jsonify(
        {
            "election_id": election.id,
            "title": election.title,
            "candidates": [
                {"candidate_id": c.id, "name": c.name, "photo_url": c.photo_url, "details": c.details}
                for c in candidates
            ],
        }
    )


@app.post("/api/voter/elections/<int:election_id>/vote")
@jwt_required()
def cast_vote(election_id: int):
    user = get_current_user()
    payload = request.get_json(force=True)
    candidate_id = payload.get("candidate_id")

    eligibility = VoterEligibility.query.filter_by(user_id=user.id, election_id=election_id).first()
    if not eligibility:
        return jsonify({"error": "Not eligible for this election"}), 403
    if eligibility.has_voted:
        return jsonify({"error": "Vote already cast"}), 409

    candidate = Candidate.query.filter_by(id=candidate_id, election_id=election_id).first()
    if not candidate:
        return jsonify({"error": "Invalid candidate"}), 400

    vote = Vote(election_id=election_id, candidate_id=candidate.id, voter_id=user.id)
    db.session.add(vote)
    db.session.flush()
    eligibility.has_voted = True
    block_hash = append_vote_block(vote)
    log_action("vote_cast", f"Vote {vote.id} recorded with block {block_hash}", user.id)
    db.session.commit()
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
@jwt_required()
def pending_registrations():
    admin_required()
    users = User.query.filter_by(registration_status="pending").all()
    return jsonify(
        [
            {
                "user_id": u.id,
                "name": u.name,
                "unique_id": u.unique_id,
                "email": u.email,
                "created_at": u.created_at.isoformat(),
            }
            for u in users
        ]
    )


@app.put("/api/admin/users/<int:user_id>/registration-status")
@jwt_required()
def registration_status(user_id: int):
    admin = admin_required()
    payload = request.get_json(force=True)
    status_value = payload.get("registration_status")
    if status_value not in {"approved", "rejected", "pending"}:
        return jsonify({"error": "registration_status must be approved, rejected, or pending"}), 400
    user = User.query.get_or_404(user_id)
    user.registration_status = status_value
    log_action("registration_status_updated", f"Set user {user.id} to {status_value}", admin.id)
    db.session.commit()
    return jsonify({"message": "Registration status updated", "user_id": user.id, "registration_status": user.registration_status})


@app.post("/api/admin/elections")
@jwt_required()
def create_election():
    admin = admin_required()
    payload = request.get_json(force=True)
    try:
        election = Election(
            title=payload["title"],
            description=payload.get("description"),
            start_time=datetime.fromisoformat(payload["start_time"]),
            end_time=datetime.fromisoformat(payload["end_time"]),
            status=payload.get("status", "scheduled"),
            created_by=admin.id,
        )
    except (KeyError, ValueError):
        return jsonify({"error": "Invalid election payload. Required: title, start_time, end_time (ISO format)."}), 400

    db.session.add(election)
    log_action("election_created", f"Election '{election.title}' created", admin.id)
    db.session.commit()
    return jsonify({"election_id": election.id, "message": "Election created"}), 201


@app.post("/api/admin/elections/<int:election_id>/candidates")
@jwt_required()
def add_candidate(election_id: int):
    admin = admin_required()
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
    log_action("candidate_added", f"Candidate '{candidate.name}' added to election {election.id}", admin.id)
    db.session.commit()
    return jsonify({"candidate_id": candidate.id, "message": "Candidate added"}), 201


@app.post("/api/admin/elections/<int:election_id>/eligibility")
@jwt_required()
def set_eligibility(election_id: int):
    admin = admin_required()
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
            db.session.add(VoterEligibility(user_id=user.id, election_id=election_id, has_voted=False))
            created += 1

    log_action("eligibility_updated", f"Eligibility set for election {election_id} with {created} voters", admin.id)
    db.session.commit()
    return jsonify({"message": "Eligibility updated", "eligibility_count": created})


@app.get("/api/admin/elections/<int:election_id>/monitor")
@jwt_required()
def monitor_election(election_id: int):
    admin_required()
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
@jwt_required()
def audit_trail():
    admin_required()
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


with app.app_context():
    db.create_all()
    admin_unique_id = os.getenv("DEFAULT_ADMIN_ID", "admin")
    admin_email = os.getenv("DEFAULT_ADMIN_EMAIL", "admin@example.com")
    if not User.query.filter_by(unique_id=admin_unique_id).first():
        # Create a valid simple synthetic face so face auth path can work for admin if needed.
        blank = np.zeros((160, 160), dtype=np.uint8)
        cv2.circle(blank, (80, 70), 35, 180, 2)
        cv2.circle(blank, (66, 65), 3, 255, -1)
        cv2.circle(blank, (94, 65), 3, 255, -1)
        cv2.ellipse(blank, (80, 85), (18, 10), 0, 0, 180, 200, 2)

        db.session.add(
            User(
                name="System Admin",
                unique_id=admin_unique_id,
                email=admin_email,
                facial_image_data=encode_image_to_base64(blank),
                registration_status="approved",
                role="admin",
            )
        )
        db.session.commit()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)

from itertools import groupby
import os
import jwt
from jwt import PyJWTError
from datetime import datetime, timedelta
from typing import List, Optional, Dict

from fastapi import FastAPI, Depends, HTTPException, status, APIRouter
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP, Boolean, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql.expression import text
import os
import jwt
from jwt import PyJWTError
from datetime import datetime, timedelta
from typing import List, Optional, Dict

from fastapi import FastAPI, Depends, HTTPException, status, APIRouter
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP, Boolean, ForeignKey, func
from sqlalchemy.orm import declarative_base
#Base = declarative_base()

from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql.expression import text
DATABASE_URL = "sqlite:///./online_voting_system.db"

# JWT Settings
SECRET_KEY = "a_very_secret_key_for_jwt_tokens"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# --- Database Setup (database.py) ---
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- SQLAlchemy Models (models.py) ---
class User(Base):
    __tablename__ = "Users"
    user_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    unique_id = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    facial_image_data = Column(Text, nullable=False)
    registration_status = Column(String, nullable=False, default='pending')
    role = Column(String, nullable=False) # 'voter' or 'admin'
    created_at = Column(TIMESTAMP, nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    created_elections = relationship("Election", back_populates="creator")

class Election(Base):
    __tablename__ = "Elections"
    election_id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text)
    start_time = Column(TIMESTAMP, nullable=False)
    end_time = Column(TIMESTAMP, nullable=False)
    status = Column(String, nullable=False, index=True) # 'scheduled', 'ongoing', 'completed', 'results_published'
    created_by = Column(Integer, ForeignKey('Users.user_id'))
    created_at = Column(TIMESTAMP, nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    creator = relationship("User", back_populates="created_elections")
    candidates = relationship("Candidate", back_populates="election", cascade="all, delete-orphan")
    votes = relationship("Vote", back_populates="election", cascade="all, delete-orphan")

class Candidate(Base):
    __tablename__ = "Candidates"
    candidate_id = Column(Integer, primary_key=True, index=True)
    election_id = Column(Integer, ForeignKey('Elections.election_id'), nullable=False)
    name = Column(String, nullable=False)
    photo_url = Column(String)
    details = Column(Text)
    election = relationship("Election", back_populates="candidates")

class VoterEligibility(Base):
    __tablename__ = "VoterEligibility"
    user_id = Column(Integer, ForeignKey('Users.user_id'), primary_key=True)
    election_id = Column(Integer, ForeignKey('Elections.election_id'), primary_key=True)
    has_voted = Column(Boolean, nullable=False, default=False)

class Vote(Base):
    __tablename__ = "Votes"
    vote_id = Column(Integer, primary_key=True, index=True)
    election_id = Column(Integer, ForeignKey('Elections.election_id'), nullable=False)
    candidate_id = Column(Integer, ForeignKey('Candidates.candidate_id'), nullable=False)
    cast_at = Column(TIMESTAMP, nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    election = relationship("Election", back_populates="votes")

class AuditTrail(Base):
    __tablename__ = "AuditTrail"
    log_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('Users.user_id'), nullable=True)
    action = Column(String, nullable=False)
    details = Column(Text)
    timestamp = Column(TIMESTAMP, nullable=False, server_default=text('CURRENT_TIMESTAMP'))

# --- Pydantic Schemas (schemas.py) ---

# User Schemas
class UserRegister(BaseModel):
    name: str
    unique_id: str
    email: EmailStr
    facial_image_data: str

class UserRegisteredResponse(BaseModel):
    user_id: int
    registration_status: str
    message: str

class UserProfile(BaseModel):
    user_id: int
    name: str
    unique_id: str
    email: EmailStr
    registration_status: str
    role: str

    class Config:
        from_attributes = True


# Auth Schemas
class LoginFace(BaseModel):
    unique_id: str
    live_facial_image_data: str

class LoginOTPRequest(BaseModel):
    email: EmailStr

class LoginOTPVerify(BaseModel):
    email: EmailStr
    otp_code: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_role: str

class TokenData(BaseModel):
    unique_id: Optional[str] = None

# Voter Schemas
class VoterElection(BaseModel):
    election_id: int
    title: str
    description: Optional[str]
    start_time: datetime
    end_time: datetime
    has_voted: bool

class CandidateInBallot(BaseModel):
    candidate_id: int
    name: str
    photo_url: Optional[str]
    details: Optional[str]

    class Config:
        from_attributes = True

class ElectionBallot(BaseModel):
    election_id: int
    title: str
    candidates: List[CandidateInBallot]

class VoteCreate(BaseModel):
    candidate_id: int

class VoteConfirmation(BaseModel):
    vote_id: int
    message: str
    cast_at: datetime

# Public Schemas
class ElectionResultCandidate(BaseModel):
    candidate_id: int
    name: str
    vote_count: int

class ElectionResult(BaseModel):
    election_id: int
    title: str
    results: List[ElectionResultCandidate]

# Admin Schemas
class PendingRegistration(BaseModel):
    user_id: int
    name: str
    unique_id: str
    email: EmailStr
    created_at: datetime

    class Config:
        from_attributes = True 

class RegistrationStatusUpdate(BaseModel):
    registration_status: str # 'approved' or 'rejected'

class RegistrationStatusResponse(BaseModel):
    user_id: int
    registration_status: str
    message: str

class ElectionCreate(BaseModel):
    title: str
    description: Optional[str]
    start_time: datetime
    end_time: datetime

class ElectionResponse(BaseModel):
    election_id: int
    title: str
    description: Optional[str]
    start_time: datetime
    end_time: datetime
    status: str
    created_by: Optional[int]

    class Config:
        from_attributes = True 

class ElectionList(BaseModel):
    election_id: int
    title: str
    start_time: datetime
    end_time: datetime
    status: str

    class Config:
        from_attributes = True 

class CandidateCreate(BaseModel):
    name: str
    photo_url: Optional[str]
    details: Optional[str]

class CandidateResponse(BaseModel):
    candidate_id: int
    election_id: int
    name: str
    photo_url: Optional[str]

    class Config:
        from_attributes = True 

class EligibilityUpdate(BaseModel):
    user_ids: List[int]

class EligibilityResponse(BaseModel):
    message: str
    eligibility_count: int

class ElectionMonitor(BaseModel):
    election_id: int
    total_eligible_voters: int
    votes_cast: int
    turnout_percentage: float

class ElectionStatusUpdate(BaseModel):
    status: str # e.g., 'published'

class ElectionStatusResponse(BaseModel):
    election_id: int
    status: str
    message: str

class AuditLog(BaseModel):
    log_id: int
    user_id: Optional[int]
    action: str
    details: Optional[str]
    timestamp: datetime

    class Config:
        from_attributes = True 

# --- Authentication (auth.py) ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login-face") # Placeholder URL

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        unique_id: str = payload.get("sub")
        if unique_id is None:
            raise credentials_exception
        token_data = TokenData(unique_id=unique_id)
    except PyJWTError:
        raise credentials_exception
    user = db.query(User).filter(User.unique_id == token_data.unique_id).first()
    if user is None:
        raise credentials_exception
    if user.registration_status != 'approved':
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User registration not approved")
    return user

def get_current_admin_user(current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
    return current_user

# --- Helper function to create audit logs ---
def create_audit_log(db: Session, user_id: Optional[int], action: str, details: str):
    log_entry = AuditTrail(user_id=user_id, action=action, details=details)
    db.add(log_entry)
    db.commit()

# --- API Routers ---
auth_router = APIRouter(prefix="/api/auth", tags=["Authentication"])
user_router = APIRouter(prefix="/api", tags=["Users & Voters"])
admin_router = APIRouter(prefix="/api/admin", tags=["Administration"], dependencies=[Depends(get_current_admin_user)])
public_router = APIRouter(prefix="/api", tags=["Public"])

# --- Authentication Endpoints ---
@auth_router.post("/register", response_model=UserRegisteredResponse, status_code=status.HTTP_201_CREATED)
def register_user(user: UserRegister, db: Session = Depends(get_db)):
    db_user_email = db.query(User).filter(User.email == user.email).first()
    if db_user_email:
        raise HTTPException(status_code=400, detail="Email already registered")
    db_user_id = db.query(User).filter(User.unique_id == user.unique_id).first()
    if db_user_id:
        raise HTTPException(status_code=400, detail="Unique ID already registered")

    new_user = User(
        name=user.name,
        unique_id=user.unique_id,
        email=user.email,
        facial_image_data=user.facial_image_data,
        role='voter', # Default role is voter
        registration_status='pending'
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    create_audit_log(db, new_user.user_id, 'user_register', f'User {new_user.unique_id} registered.')
    return {"user_id": new_user.user_id, "registration_status": "pending", "message": "Registration successful, awaiting admin approval."}

@auth_router.post("/login-face", response_model=Token)
def login_face(form_data: LoginFace, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.unique_id == form_data.unique_id).first()
    if not user or user.registration_status != 'approved':
        raise HTTPException(status_code=401, detail="Authentication failed or user not approved")

    # In a real app, facial comparison logic would be here.
    # We simulate a successful facial match.

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.unique_id}, expires_delta=access_token_expires
    )
    create_audit_log(db, user.user_id, 'user_login', f'User {user.unique_id} logged in via face.')
    return {"access_token": access_token, "user_role": user.role}

@auth_router.post("/login-otp-request")
def request_otp(form_data: LoginOTPRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.email).first()
    if not user or user.registration_status != 'approved':
        raise HTTPException(status_code=404, detail="No approved user found with this email")
    # In a real app, generate and send OTP via email. We'll just simulate.
    # e.g., store otp in cache with user's email as key
    print(f"OTP for {user.email}: 123456 (simulation)")
    return {"message": "An OTP has been sent to your registered email address."}

@auth_router.post("/login-otp-verify", response_model=Token)
def verify_otp(form_data: LoginOTPVerify, db: Session = Depends(get_db)):
    # In a real app, verify OTP from cache/db
    if form_data.otp_code != "123456": # Simulated OTP
        raise HTTPException(status_code=400, detail="Invalid OTP")
    user = db.query(User).filter(User.email == form_data.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.unique_id}, expires_delta=access_token_expires
    )
    create_audit_log(db, user.user_id, 'user_login', f'User {user.unique_id} logged in via OTP.')
    return {"access_token": access_token, "user_role": user.role}

# --- User and Voter Endpoints ---
@user_router.get("/users/me", response_model=UserProfile)
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@user_router.get("/voter/elections", response_model=List[VoterElection])
def get_voter_elections(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    now = datetime.utcnow()
    eligible_elections = db.query(Election, VoterEligibility.has_voted).\
        join(VoterEligibility, VoterEligibility.election_id == Election.election_id).\
        filter(VoterEligibility.user_id == current_user.user_id).\
        filter(Election.status == 'ongoing').\
        filter(Election.start_time <= now).\
        filter(Election.end_time >= now).all()

    result = []
    for election, has_voted in eligible_elections:
        result.append(VoterElection(
            election_id=election.election_id,
            title=election.title,
            description=election.description,
            start_time=election.start_time,
            end_time=election.end_time,
            has_voted=has_voted
        ))
    return result

@user_router.get("/voter/elections/{election_id}/ballot", response_model=ElectionBallot)
def get_election_ballot(election_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    election = db.query(Election).filter(Election.election_id == election_id).first()
    if not election:
        raise HTTPException(status_code=404, detail="Election not found")
    return ElectionBallot(election_id=election.election_id, title=election.title, candidates=election.candidates)

@user_router.post("/voter/elections/{election_id}/vote", response_model=VoteConfirmation)
def cast_vote(election_id: int, vote: VoteCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    now = datetime.utcnow()
    eligibility = db.query(VoterEligibility).\
        filter(VoterEligibility.user_id == current_user.user_id, VoterEligibility.election_id == election_id).first()
    if not eligibility:
        raise HTTPException(status_code=403, detail="You are not eligible to vote in this election")
    if eligibility.has_voted:
        raise HTTPException(status_code=403, detail="You have already voted in this election")
    
    election = db.query(Election).filter(Election.election_id == election_id).first()
    if not election or election.status != 'ongoing' or not (election.start_time <= now <= election.end_time):
        raise HTTPException(status_code=403, detail="This election is not currently active")

    candidate = db.query(Candidate).filter(Candidate.candidate_id == vote.candidate_id, Candidate.election_id == election_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found for this election")

    new_vote = Vote(election_id=election_id, candidate_id=vote.candidate_id)
    eligibility.has_voted = True
    db.add(new_vote)
    db.commit()
    db.refresh(new_vote)
    create_audit_log(db, current_user.user_id, 'vote_cast', f'Voted for candidate {candidate.name} in election {election.title}')
    return {"vote_id": new_vote.vote_id, "message": "Your vote has been successfully cast.", "cast_at": new_vote.cast_at}

# --- Public Endpoints ---
@public_router.get("/elections/{election_id}/results", response_model=ElectionResult)
def get_election_results(election_id: int, db: Session = Depends(get_db)):
    election = db.query(Election).filter(Election.election_id == election_id).first()
    if not election:
        raise HTTPException(status_code=404, detail="Election not found")
    if election.status not in ['completed', 'results_published']:
         raise HTTPException(status_code=403, detail="Results for this election are not yet public")

    results = db.query(
        Candidate.candidate_id, 
        Candidate.name, 
        func.count(Vote.vote_id).label('vote_count')
    ).join(Vote, Candidate.candidate_id == Vote.candidate_id, isouter=True).\
    filter(Candidate.election_id == election_id).\
    group_by(Candidate.candidate_id, Candidate.name).\
    order_by(func.count(Vote.vote_id).desc()).all()

    return {
        "election_id": election.election_id,
        "title": election.title,
        "results": results
    }

# --- Admin Endpoints ---
@admin_router.get("/registrations", response_model=List[PendingRegistration])
def get_pending_registrations(status: str = 'pending', db: Session = Depends(get_db)):
    users = db.query(User).filter(User.registration_status == status).all()
    return users

@admin_router.put("/registrations/{user_id}/status", response_model=RegistrationStatusResponse)
def update_registration_status(user_id: int, status_update: RegistrationStatusUpdate, current_admin: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if status_update.registration_status not in ['approved', 'rejected']:
        raise HTTPException(status_code=400, detail="Invalid status")
    
    user.registration_status = status_update.registration_status
    db.commit()
    create_audit_log(db, current_admin.user_id, 'registration_update', f'Set registration status of user {user.unique_id} to {user.registration_status}')
    return {"user_id": user.user_id, "registration_status": user.registration_status, "message": f"User status updated to {user.registration_status}"}

@admin_router.post("/elections", response_model=ElectionResponse, status_code=status.HTTP_201_CREATED)
def create_election(election: ElectionCreate, current_admin: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    new_election = Election(**election.dict(), created_by=current_admin.user_id, status='scheduled')
    db.add(new_election)
    db.commit()
    db.refresh(new_election)
    create_audit_log(db, current_admin.user_id, 'election_created', f'Created election: {new_election.title}')
    return new_election

@admin_router.get("/elections", response_model=List[ElectionList])
def get_all_elections(status: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(Election)
    if status:
        query = query.filter(Election.status == status)
    return query.all()

@admin_router.put("/elections/{election_id}", response_model=ElectionResponse)
def update_election(election_id: int, election_update: ElectionCreate, current_admin: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    election = db.query(Election).filter(Election.election_id == election_id).first()
    if not election:
        raise HTTPException(status_code=404, detail="Election not found")
    if election.status != 'scheduled':
        raise HTTPException(status_code=403, detail="Can only update elections that are scheduled")

    election.title = election_update.title
    election.description = election_update.description
    election.start_time = election_update.start_time
    election.end_time = election_update.end_time
    db.commit()
    db.refresh(election)
    create_audit_log(db, current_admin.user_id, 'election_updated', f'Updated election: {election.title}')
    return election

@admin_router.post("/elections/{election_id}/candidates", response_model=CandidateResponse, status_code=status.HTTP_201_CREATED)
def add_candidate(election_id: int, candidate: CandidateCreate, current_admin: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    election = db.query(Election).filter(Election.election_id == election_id).first()
    if not election:
        raise HTTPException(status_code=404, detail="Election not found")
    
    new_candidate = Candidate(**candidate.dict(), election_id=election_id)
    db.add(new_candidate)
    db.commit()
    db.refresh(new_candidate)
    create_audit_log(db, current_admin.user_id, 'candidate_added', f'Added candidate {new_candidate.name} to election {election.title}')
    return new_candidate

@admin_router.delete("/elections/{election_id}/candidates/{candidate_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_candidate(election_id: int, candidate_id: int, current_admin: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    candidate = db.query(Candidate).filter(Candidate.election_id == election_id, Candidate.candidate_id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    db.delete(candidate)
    db.commit()
    create_audit_log(db, current_admin.user_id, 'candidate_removed', f'Removed candidate ID {candidate_id} from election ID {election_id}')
    return {"message": "Candidate removed successfully"}

@admin_router.post("/elections/{election_id}/eligibility", response_model=EligibilityResponse)
def set_voter_eligibility(election_id: int, eligibility: EligibilityUpdate, current_admin: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    election = db.query(Election).filter(Election.election_id == election_id).first()
    if not election:
        raise HTTPException(status_code=404, detail="Election not found")
    
    # Clear existing eligibility for this election to prevent duplicates on re-run
    db.query(VoterEligibility).filter(VoterEligibility.election_id == election_id).delete()

    eligible_users = db.query(User).filter(User.user_id.in_(eligibility.user_ids), User.registration_status == 'approved').all()
    count = 0
    for user in eligible_users:
        eligibility_entry = VoterEligibility(user_id=user.user_id, election_id=election_id)
        db.add(eligibility_entry)
        count += 1
    db.commit()
    create_audit_log(db, current_admin.user_id, 'eligibility_set', f'Set eligibility for {count} voters in election {election.title}')
    return {"message": f"Successfully set eligibility for {count} voters.", "eligibility_count": count}

@admin_router.get("/elections/{election_id}/monitor", response_model=ElectionMonitor)
def monitor_election(election_id: int, db: Session = Depends(get_db)):
    total_eligible = db.query(VoterEligibility).filter(VoterEligibility.election_id == election_id).count()
    votes_cast = db.query(Vote).filter(Vote.election_id == election_id).count()
    turnout = (votes_cast / total_eligible * 100) if total_eligible > 0 else 0
    return {
        "election_id": election_id,
        "total_eligible_voters": total_eligible,
        "votes_cast": votes_cast,
        "turnout_percentage": round(turnout, 2)
    }

@admin_router.put("/elections/{election_id}/status", response_model=ElectionStatusResponse)
def update_election_status(election_id: int, status_update: ElectionStatusUpdate, current_admin: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    election = db.query(Election).filter(Election.election_id == election_id).first()
    if not election:
        raise HTTPException(status_code=404, detail="Election not found")
    election.status = status_update.status
    db.commit()
    create_audit_log(db, current_admin.user_id, 'election_status_update', f'Updated status for election {election.title} to {election.status}')
    return {"election_id": election.election_id, "status": election.status, "message": f"Election status updated to {election.status}"}

@admin_router.get("/audittrail", response_model=List[AuditLog])
def get_audit_trail(user_id: Optional[int] = None, action: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(AuditTrail)
    if user_id:
        query = query.filter(AuditTrail.user_id == user_id)
    if action:
        query = query.filter(AuditTrail.action == action)
    return query.order_by(AuditTrail.timestamp.desc()).all()


# --- Main FastAPI App ---
app = FastAPI(title="Online Voting System API")

# Include routers
app.include_router(auth_router)
app.include_router(user_router)
app.include_router(admin_router)
app.include_router(public_router)

from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create database tables
    Base.metadata.create_all(bind=engine)

    # Create a default admin user if one doesn't exist
    db = SessionLocal()
    admin = db.query(User).filter(User.role == 'admin').first()
    if not admin:
        admin_user = User(
            name='Admin User',
            unique_id='admin001',
            email='admin@example.com',
            facial_image_data='base64encodedadmindata',
            registration_status='approved',
            role='admin'
        )
        db.add(admin_user)
        db.commit()
        print("Default admin user created.")
    db.close()

    # startup completed
    yield

    # (Optional) Shutdown cleanup code likhna ho to yaha add karo
    print("App is shutting down...")

# App initialization
app = FastAPI(lifespan=lifespan)

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Online Voting System API"}
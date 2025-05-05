import os
import json
import whisper
import ffmpeg
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Whisper model once
try:
    MODEL = whisper.load_model("base")
    logger.info("Loaded Whisper model 'base'.")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    MODEL = None

# Local Python-based video → audio → timestamped transcript using ffmpeg-python


import os
import json
import whisper
import ffmpeg
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Whisper model once
try:
    MODEL = whisper.load_model("base")
    logger.info("Loaded Whisper model 'base'.")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    MODEL = None

def process_video(video_path: str):
    """
    1. Extracts audio (WAV) from the given video file using ffmpeg-python.
    2. Runs Whisper locally to produce a simple transcript (no timestamps).
    3. Saves the audio in /audio and the transcript text in /transcripts (as JSON with a 'text' key).

    Returns:
        audio_path (str)
        transcript_path (str)
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    base_dir = os.getcwd()
    audio_dir = os.path.join(base_dir, 'audio')
    transcript_dir = os.path.join(base_dir, 'transcripts')
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(transcript_dir, exist_ok=True)

    # 1. Extract audio
    audio_path = os.path.join(audio_dir, f"{base_name}.wav")
    logger.info(f"Extracting audio to {audio_path}...")
    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        logger.info("Audio extraction complete.")
    except ffmpeg.Error as e:
        err = e.stderr.decode() if hasattr(e, 'stderr') else str(e)
        logger.error(f"ffmpeg extraction error: {err}")
        raise

    if not os.path.exists(audio_path):
        msg = f"Audio not found at {audio_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    
    print("Step 1: Starting video processing...")

    print("Step 2: Audio extracted to:", audio_path)

    # 2. Simple Whisper transcription
    if MODEL is None:
        raise RuntimeError("Whisper model not loaded.")
    logger.info("Running simple transcription (no timestamps)...")
    try:
        result = MODEL.transcribe(audio_path)  # returns {'text': ..., ...}
        transcript_text = result.get('text', '').strip()
        logger.info("Transcription complete.")
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise

    # 3. Save transcript
    transcript_path = os.path.join(transcript_dir, f"{base_name}.json")
    try:
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump({"text": transcript_text}, f, ensure_ascii=False, indent=2)
        logger.info(f"Transcript saved to {transcript_path}.")
    except Exception as e:
        logger.error(f"Failed to save transcript: {e}")
        raise
    print("Step 3: Transcript written to:", transcript_path)    
    return audio_path, transcript_path


import cv2
import mediapipe as mp
import time
from datetime import datetime

# Initialize MediaPipe models
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

class InterviewAnalytics:
    def __init__(self):
        self.session_start_time = time.time()
        self.eye_contact_duration = 0
        self.last_eye_contact_time = None
        self.hand_gesture_counts = {
            "open_palm": 0,
            "closed_fist": 0,
            "pointing": 0,
            "hand_near_face": 0,
            "excessive_movement": 0,
            "neutral": 0,
            "thumbs_up": 0
        }
        self.poor_posture_duration = 0
        self.last_poor_posture_time = None

    def update_eye_contact(self, has_contact):
        current_time = time.time()
        if has_contact:
            if self.last_eye_contact_time is not None:
                self.eye_contact_duration += (current_time - self.last_eye_contact_time)
            self.last_eye_contact_time = current_time
        else:
            self.last_eye_contact_time = None

    def update_gesture(self, gesture_type):
        if gesture_type in self.hand_gesture_counts:
            self.hand_gesture_counts[gesture_type] += 1

    def update_posture(self, is_good_posture):
        current_time = time.time()
        if not is_good_posture:
            if self.last_poor_posture_time is not None:
                self.poor_posture_duration += (current_time - self.last_poor_posture_time)
            self.last_poor_posture_time = current_time
        else:
            self.last_poor_posture_time = None

    def get_session_duration(self):
        return time.time() - self.session_start_time

    def get_eye_contact_percentage(self):
        duration = self.get_session_duration()
        return (self.eye_contact_duration / duration) * 100 if duration > 0 else 0

    def get_poor_posture_percentage(self):
        duration = self.get_session_duration()
        return (self.poor_posture_duration / duration) * 100 if duration > 0 else 0

    def get_dominant_gesture(self):
        return max(self.hand_gesture_counts, key=self.hand_gesture_counts.get)
    
    def _compute_gesture_score(self):
        total = sum(self.analytics.hand_gesture_counts.values())
        return int(min((total / 20) * 100, 100))  


class InterviewAnalyzer:
    def __init__(self):
        self.analytics = InterviewAnalytics()
        self.hands_model = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        self.face_model = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
        self.pose_model = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    def analyze_frame(self, image):
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        hand_results = self.hands_model.process(image_rgb)
        face_results = self.face_model.process(image_rgb)
        pose_results = self.pose_model.process(image_rgb)

        # Dummy logic
        if hand_results.multi_hand_landmarks:
            self.analytics.update_gesture("open_palm")
        if face_results.multi_face_landmarks:
            self.analytics.update_eye_contact(True)
        if pose_results.pose_landmarks:
            self.analytics.update_posture(True)

    def generate_report(self):
        report = {
            "timestamp": datetime.now().isoformat(),
            "duration": int(self.analytics.get_session_duration()),
            "eye_contact": int(self.analytics.get_eye_contact_percentage()),
            "posture": 100 - int(self.analytics.get_poor_posture_percentage()),
            "hand_gesture_score": self.analytics._compute_gesture_score(),
            "gestures": {g: c for g, c in self.analytics.hand_gesture_counts.items() if c > 0},
            "recommendations": self._generate_recommendations()
        }
        return report

    def _generate_recommendations(self):
        recs = []
        if self.analytics.get_eye_contact_percentage() < 60:
            recs.append("Improve eye contact")
        if self.analytics.get_poor_posture_percentage() > 30:
            recs.append("Maintain better posture")
        if self.analytics.hand_gesture_counts["hand_near_face"] > 5:
            recs.append("Avoid touching face during interviews")
        if self.analytics.hand_gesture_counts["excessive_movement"] > 10:
            recs.append("Reduce hand movement to show confidence")
        return recs

def coach_video_file(path):
    cap = cv2.VideoCapture(path)
    analyzer = InterviewAnalyzer()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        analyzer.analyze_frame(frame)

    cap.release()
    return analyzer.generate_report()
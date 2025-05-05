import os
import json
import logging
import numpy as np
import librosa
import cohere
from video_processing import coach_video_file


# API Key (ideally set via env var)
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "O34WBadHOatc1tlLhoHnkLNx8Ov2nfU0MOgaa1Sy")

# ------------------- COHERE FEEDBACK GENERATOR -------------------

class FeedbackGenerator:
    def __init__(self, api_key):
        self.client = cohere.Client(api_key)
        

    def generate(self, question, transcript, prosody, behavior_analysis):
        gesture_summary = "\n".join([
            f"- {k.replace('_', ' ').title()}: {v}" 
            for k, v in behavior_analysis.get("gestures", {}).items()
        ])
        recommendations = "\n".join([
            f"- {r}" 
            for r in behavior_analysis.get("recommendations", [])
        ])

        prompt = f"""
You are an expert AI interview coach. You will evaluate a candidate’s response using both content (transcript), delivery (prosody features: speed, pitch, energy), and body language (eye contact, posture, hand gestures).

---

QUESTION: {question}

TRANSCRIPT:
{transcript}

VOICE ANALYSIS:
- Speaking rate: {prosody['speaking_rate']:.1f} wpm
- Pitch variation (std): {prosody['pitch_stats']['std']:.2f}
- Volume variation (std): {prosody['energy_stats']['std']:.2f}
- Duration: {prosody['duration']:.1f} seconds

---
BODY LANGUAGE:
- Eye contact: {behavior_analysis.get('eye_contact', 0)}%
- Posture score: {behavior_analysis.get('posture', 0)}%
- Hand gestures:
{gesture_summary}

RECOMMENDED CHANGES FROM BODY LANGUAGE:
{recommendations}

SCORING RUBRIC (0–10):

VOCAL DELIVERY

SPEAKING RATE:
- < 120 WPM: score 3–4 (too slow, hesitant)
- 120–140 WPM: score 5–6 (somewhat slow but acceptable)
- 140–160 WPM: score 8–9 (ideal)
- 160–180 WPM: score 6–7 (slightly fast but clear)
- > 180 WPM: score 3–5 (rushed, hard to follow)

PITCH STANDARD DEVIATION:
- < 30: score 3–4 (monotone, lacks expression)
- 30–80: score 5–6 (limited variation)
- 80–120: score 8–9 (natural expressiveness)
- > 120: score 6–7 (possibly erratic)

ENERGY STANDARD DEVIATION:
- < 0.02: score 3–4 (flat, low projection)
- 0.02–0.06: score 5–6 (mild variation)
- 0.06–0.10: score 8–9 (good projection)
- > 0.10: score 6–7 (possibly inconsistent)

🎯 CONTENT & LANGUAGE
- Evaluate transcript for:
  - Relevance to the question
  - Completeness of response
  - Clarity & conciseness
  - Vocabulary richness
  - Verbal fluency (e.g., filler words, disfluencies)

---

Please return feedback in this JSON format:
{{
  "overall_score": 0-10,
  "content_language": {{
    "score": 0-10,
    "relevance": {{ "score": 0-10, "feedback": "" }},
    "completeness": {{ "score": 0-10, "feedback": "" }},
    "clarity_conciseness": {{ "score": 0-10, "feedback": "" }},
    "lexical_richness": {{ "score": 0-10, "feedback": "" }},
    "fluency": {{ "score": 0-10, "feedback": "" }}
  }},
  "vocal_delivery": {{
    "score": 0-10,
    "pace": {{ "score": 0-10, "feedback": "" }},
    "intonation_emphasis": {{ "score": 0-10, "feedback": "" }},
    "volume_projection": {{ "score": 0-10, "feedback": "" }},
    "confidence_signals": {{ "score": 0-10, "feedback": "" }}
  }},
  "strengths": ["", "", ""],
  "improvement_areas": ["", "", ""],
  "summary_feedback": ""
}}

ONLY return the JSON. Do NOT explain anything else. Ensure scores vary and are justified.
"""
        print("Sending prompt to Cohere...")
        print(prompt)
        response = self.client.generate(
            prompt=prompt,
            model="command-r-plus",
            max_tokens=2048,
            temperature=0.2,
            stop_sequences=[],
            return_likelihoods="NONE"
        )
       
        print("Raw response:")
        print(response.generations[0].text)

        try:
            return json.loads(response.generations[0].text)
        except json.JSONDecodeError:
            return {
                "error": "LLM output could not be parsed.",
                "raw_output": response.generations[0].text
            }

# ------------------- AUDIO ANALYSIS -------------------

def analyze_audio(audio_path: str) -> dict:
    y, sr = librosa.load(audio_path, sr=16000)
    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)

    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        frame_length=frame_len,
        hop_length=hop_len
    )
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_len)
    onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=hop_len)

    return {
        'rms': rms.tolist(),
        'pitch': [float(x) if not np.isnan(x) else None for x in f0],
        'voiced_flag': voiced_flag.tolist(),
        'voiced_prob': voiced_prob.tolist(),
        'onset_times': onset_times.tolist(),
        'speaking_rate': len(onset_times) / (len(rms) * (10 / 1000)),
        'pitch_stats': {
            'mean': float(np.nanmean(f0)),
            'std': float(np.nanstd(f0)),
            'range': float(np.nanmax(f0) - np.nanmin(f0))
        },
        'energy_stats': {
            'mean': float(np.mean(rms)),
            'std': float(np.std(rms))
        },
        'duration': librosa.get_duration(y=y, sr=sr)
    }

# ------------------- TRANSCRIPT UTILITY -------------------

def load_transcript(transcript_path: str) -> list:
    with open(transcript_path, 'r') as f:
        data = json.load(f)
    if 'segments' not in data and 'text' in data:
        return [{'start': 0.0, 'end': 10.0, 'text': data['text']}]
    return [{'start': seg['start'], 'end': seg['end'], 'text': seg['text']} 
            for seg in data.get('segments', [])]

def normalize(x):
    if isinstance(x, (int, float)):
        return int(round(min(max(x, 0), 100)))  # cap at 100, but don't scale 0-1 to 0-100
    return 0

# ------------------- FEEDBACK STORAGE -------------------

def save_feedback(feedback: dict, out_dir: str = 'feedback') -> str:
    os.makedirs(out_dir, exist_ok=True)
    base = feedback.get('video', 'output')
    out_path = os.path.join(out_dir, f"{base}_feedback.json")
    with open(out_path, 'w') as f:
        json.dump(feedback, f, indent=2)
    logger.info(f"Feedback saved to {out_path}")
    return out_path

# ------------------- REMOVE THIS (COMMENTED OUT) -------------------

# def cohere_process_feedback(transcript_segments, audio_features, behavior_analysis):
#     # 🔴 Deprecated: replaced by FeedbackGenerator.generate()
#     # Left here for reference or fallback use only.
#     ...
#     return {
#         "summary": summary,
#         "metrics": {
#             "confidence": confidence,
#             "clarity": clarity,
#             "relevance": relevance
#         }
#     }

# ------------------- LOGGING + INSTANCE -------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
feedback_model = FeedbackGenerator(COHERE_API_KEY)

import os
import json
from datetime import datetime
import cohere
from flask import (
    Flask, request, redirect, url_for, session,
    render_template, abort, jsonify, send_from_directory
)
from video_processing import process_video, coach_video_file
from Au_trans_feat_extract import analyze_audio, load_transcript, save_feedback, feedback_model

# Cohere setup
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "O34WBadHOatc1tlLhoHnkLNx8Ov2nfU0MOgaa1Sy")
co = cohere.Client(COHERE_API_KEY)

# Resume extraction dependencies
try:
    import PyPDF2
    resume_extraction_available = True
except ImportError:
    PyPDF2 = None
    docx = None
    resume_extraction_available = False

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "averylongandsecuresecretforthisapplication")
users = {"user": "llm_password@001"}

def extract_text_from_resume(file):
    if not resume_extraction_available:
        return "Extraction unavailable"
    name = file.filename.lower()
    if name.endswith('.pdf'):
        temp = 'temp_resume.pdf'; file.save(temp)
        try:
            reader = PyPDF2.PdfReader(temp)
            return ''.join(page.extract_text() or '' for page in reader.pages)
        finally:
            os.remove(temp)
    if name.endswith('.docx'):
        temp = 'temp_resume.docx'; file.save(temp)
        try:
            doc = docx.Document(temp)
            return '\n'.join(p.text for p in doc.paragraphs)
        finally:
            os.remove(temp)
    return "Unsupported format"

def generate_questions_with_cohere(prompt: str) -> list[str]:
    res = co.generate(prompt=prompt, max_tokens=300, temperature=0.9, model="command")
    content = res.generations[0].text
    return [l.strip() for l in content.split('\n') if l.strip()]

def generate_interview_questions(job_description="", resume_text="", prompt_type="technical"):
    pt = prompt_type.lower()
    if pt == "technical":
        role, focus, base = (
            "expert technical recruiter",
            "technical questions that drill into the required skills and experience",
            job_description
        )
    elif pt == "behavioral":
        role, focus, base = (
            "expert behavioral interviewer",
            "behavioral questions to assess soft‑skills, culture and fit",
            job_description
        )
    elif pt == "resume-based":
        role, focus, base = (
            "expert interviewer reviewing candidate resumes",
            "questions that probe the candidate's past experiences and achievements",
            resume_text
        )
    else:
        raise ValueError("Invalid prompt_type")
    if not base:
        raise ValueError(f"{prompt_type} input required.")
    prompt = f"You are an {role}. Based on the following input: \"{base}\", generate exactly one interview question that {focus}. Respond only with the question and nothing else."
    return generate_questions_with_cohere(prompt)

def extract_form_data(req):
    text = ""
    if 'resume' in req.files and req.files['resume'].filename:
        text = extract_text_from_resume(req.files['resume'])
    return (
        req.form.get('interview-type'),
        req.form.get('job-role'),
        req.form.get('job-description-text', ''),
        text
    )

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        u = request.form.get('username'); p = request.form.get('password')
        if u in users and users[u] == p:
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        return "Invalid credentials"
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if session.get('logged_in'):
        return render_template('dashboard.html')
    return redirect(url_for('login'))

@app.route('/interview', methods=['POST'])
def interview():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    it, role, jd, rt = extract_form_data(request)
    if it == 'resume-based' and not rt:
        return render_template('error.html', message="Resume required.")
    if it in ['technical', 'behavioral'] and not jd:
        return render_template('error.html', message="Job description required.")
    try:
        qs = generate_interview_questions(jd, rt, it)
    except Exception as e:
        return render_template('error.html', message=str(e))
    if not qs:
        return render_template('error.html', message="No questions generated.")
    session['current_question'] = qs[0]
    return render_template('interview.html', question=qs[0])

@app.route('/results')
def results():
    feedback_dir = os.path.join(app.root_path, 'feedback')
    feedback_files = [f for f in os.listdir(feedback_dir) if f.endswith('.json')]
    if not feedback_files:
        return render_template('error.html', message="No feedback found.")

    latest_file = max(feedback_files, key=lambda f: os.path.getmtime(os.path.join(feedback_dir, f)))
    with open(os.path.join(feedback_dir, latest_file), 'r') as f:
        raw = json.load(f)
        feedback = raw.get("feedback", raw)

        if 'cohere_feedback' not in feedback and 'metrics' not in feedback['cohere_feedback']:
            feedback['cohere_feedback']['metrics'] = {
                "metrics":{
                    "confidence": 0,
                    "clarity": 0,
                    "relevance": 0
                 },
            "summary": "Feedback unavailable."
            }

    return render_template('results.html', feedback=feedback)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.route('/save_video', methods=['POST'])
def save_video():
    if 'video' not in request.files or not request.files['video'].filename:
        return jsonify({"error": "No video provided."}), 400

    vid = request.files['video']
    vid_dir = os.path.join(app.root_path, 'videos')
    os.makedirs(vid_dir, exist_ok=True)
    fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.webm"
    vpath = os.path.join(vid_dir, fname)
    vid.save(vpath)

    try:
        print(f"Processing video at: {vpath}")
        audio_path, transcript_path = process_video(vpath)
        audio_features = analyze_audio(audio_path)
        behavioral_data = coach_video_file(vpath)

        transcript_segments = load_transcript(transcript_path)
        question_text = session.get('current_question', '')
        if question_text:
            transcript_segments.insert(0, {'start': 0.0, 'end': 0.0, 'text': f"Question: {question_text}"})

        spoken_text = " ".join([seg["text"] for seg in transcript_segments])

        feedback_text = feedback_model.generate(
            question_text,
            spoken_text,
            audio_features,
            behavioral_data
        )

        feedback = {
            "video": fname,
            "audio_path": f"/audio/{os.path.basename(audio_path)}",
            "transcript_path": f"/transcripts/{os.path.basename(transcript_path)}",
            "audio_features": audio_features,
            "transcript_segments": transcript_segments,
            "cohere_feedback": feedback_text,
            "behavioral_analysis": behavioral_data
        }

        feedback_file = save_feedback(feedback)

        return jsonify({
            "message": "Processed successfully",
            "video": fname,
            "audio": feedback["audio_path"],
            "transcript": feedback["transcript_path"],
            "feedback_file": f"/feedback/{os.path.basename(feedback_file)}",
            "feedback": feedback
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory(os.path.join(app.root_path,'audio'), filename)

@app.route('/transcripts/<path:filename>')
def serve_transcript(filename):
    return send_from_directory(os.path.join(app.root_path,'transcripts'), filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

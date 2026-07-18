import os
import json
import tempfile

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
from pypdf import PdfReader
from groq import Groq


# ==========================================
# Load Environment Variables
# ==========================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")


# ==========================================
# Initialize Groq Client
# ==========================================

client = Groq(
    api_key=GROQ_API_KEY
)


# ==========================================
# FastAPI App
# ==========================================

app = FastAPI(
    title="AI Resume Analyzer",
    version="4.0"
)


# ==========================================
# CORS Configuration
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Home Route
# ==========================================

@app.get("/")
def home():

    return {
        "message": "AI Resume Analyzer Backend is Running",
        "status": "success"
    }
# ==========================================
# PDF Text Extraction
# ==========================================

def extract_text(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text.strip()


# ==========================================
# Analyze Resume API
# ==========================================

@app.post("/analyze")
async def analyze_resume(

    resume: UploadFile = File(...),

    job_role: str = Form(...)

):

    try:

        # ----------------------------------
        # Save Uploaded PDF
        # ----------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(
                await resume.read()
            )

            pdf_path = temp_file.name


        # ----------------------------------
        # Extract Resume Text
        # ----------------------------------

        resume_text = extract_text(pdf_path)

        os.remove(pdf_path)


        if not resume_text:

            return {

                "error": "Resume is empty",

                "message": "Could not extract text from the uploaded PDF."

            }


        # Limit input size

        resume_text = resume_text[:6000]


        # ----------------------------------
        # AI Prompt
        # ----------------------------------

        prompt = f"""
You are an expert ATS Resume Analyzer.

Analyze the following resume for the target job role.

Target Job Role:
{job_role}

Resume:
{resume_text}

Return ONLY valid JSON.

Do NOT use markdown.

Return exactly in this format:

{{
    "resume_score": 85,
    "ats_score": 80,
    "strengths": [
        "Strength 1",
        "Strength 2"
    ],
    "weaknesses": [
        "Weakness 1",
        "Weakness 2"
    ],
    "missing_skills": [
        "Skill 1",
        "Skill 2"
    ],
    "suggestions": [
        "Suggestion 1",
        "Suggestion 2"
    ]
}}
"""
                # ----------------------------------
        # Groq API Call
        # ----------------------------------

        response = client.chat.completions.create(

            model="llama-3.1-8b-instant",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.3,

            max_tokens=1000

        )


        # ----------------------------------
        # Get AI Response
        # ----------------------------------

        answer = response.choices[0].message.content.strip()


        # ----------------------------------
        # Remove Markdown if Present
        # ----------------------------------

        answer = answer.replace(
            "```json",
            ""
        )

        answer = answer.replace(
            "```",
            ""
        )

        answer = answer.strip()


        # ----------------------------------
        # Convert JSON String to Dictionary
        # ----------------------------------

        result = json.loads(answer)


        return result
        # ----------------------------------
    # Invalid JSON Returned by AI
    # ----------------------------------

    except json.JSONDecodeError:

        return {

            "error": "Invalid AI Response",

            "message": "The AI returned an invalid JSON response."

        }


    # ----------------------------------
    # File Error
    # ----------------------------------

    except FileNotFoundError:

        return {

            "error": "File Error",

            "message": "Uploaded resume file could not be processed."

        }


    # ----------------------------------
    # Any Other Error
    # ----------------------------------

    except Exception as e:

        return {

            "error": "Something went wrong",

            "message": str(e)

        }
    
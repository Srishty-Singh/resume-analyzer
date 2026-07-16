import os
import json
import tempfile

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
from pypdf import PdfReader

from google import genai


# =============================
# Gemini Setup
# =============================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found in .env file"
    )


client = genai.Client(
    api_key=api_key
)



# =============================
# FastAPI Setup
# =============================

app = FastAPI(
    title="AI Resume Analyzer"
)



app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)



# =============================
# Home Route
# =============================

@app.get("/")
def home():

    return {
        "message":
        "AI Resume Analyzer Backend is Running"
    }




# =============================
# PDF Text Extraction
# =============================

def extract_text(path):

    reader = PdfReader(path)

    text = ""


    for page in reader.pages:

        page_text = page.extract_text()


        if page_text:

            text += page_text + "\n"



    return text




# =============================
# Analyze Resume API
# =============================

@app.post("/analyze")
async def analyze_resume(

    resume: UploadFile = File(...),

    job_role: str = Form(...)

):

    try:


        # Save uploaded PDF temporarily

        with tempfile.NamedTemporaryFile(

            delete=False,

            suffix=".pdf"

        ) as temp:


            temp.write(
                await resume.read()
            )


            pdf_path = temp.name




        # Extract PDF text

        resume_text = extract_text(pdf_path)



        # Delete temporary file

        os.remove(pdf_path)



        # Limit input size for faster Gemini response

        resume_text = resume_text[:6000]



        prompt = f"""

You are an ATS resume expert.

Analyze this resume according to the given job role.


Target Job Role:

{job_role}



Resume Content:

{resume_text}



Return ONLY valid JSON.

Do not add markdown.

Use this exact format:


{{
"resume_score":80,
"ats_score":75,
"strengths":[
"example strength"
],
"weaknesses":[
"example weakness"
],
"missing_skills":[
"skill 1"
],
"suggestions":[
"suggestion 1"
]
}}

"""



        # Gemini API Call

        response = client.models.generate_content(

            model="gemini-flash-latest",

            contents=prompt

        )



        answer = response.text.strip()



        # Remove markdown formatting if present

        answer = answer.replace(

            "```json",

            ""

        )


        answer = answer.replace(

            "```",

            ""

        )



        result = json.loads(answer)



        return result




    except Exception as e:


        error = str(e)



        if "429" in error:


            return {


                "error":
                "Gemini API quota exceeded",


                "message":
                "Please check Gemini API quota or billing."

            }



        return {


            "error":
            "Something went wrong",


            "message":
            error

        }
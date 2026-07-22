# 💼 Job Application Assistant

An AI-powered tool that analyses your resume against any job description and generates a complete application package — in under 30 seconds.

## What it does

Upload your resume PDF and paste a job description. The pipeline runs four AI chains and returns:

- **Skill Gap Report** — match score, missing skills with learning resources, your strengths for the role
- **3 Cover Letter Versions** — formal, conversational, and concise — each tailored to your actual experience
- **Interview Preparation** — 10 role-specific questions with suggested answers based on your resume
- **Downloadable ZIP** — all outputs bundled for offline use

## Demo

> Upload resume → paste JD → click Analyse → download your package

## Tech stack

| Layer | Technology |
|---|---|
| LLM | Llama 3.3 70B via Groq API |
| Orchestration | LangChain LCEL |
| Structured output | Pydantic v2 |
| PDF parsing | PyMuPDF |
| UI | Streamlit |
| Language | Python 3.10+ |

## How the pipeline works

```
Resume PDF + Job Description
        ↓
Step 1 — Parallel (simultaneous):
   ├── JD Analysis      → extracts skills, requirements, culture signals
   └── Resume Analysis  → extracts skills, projects, achievements

Step 2 — Skill Gap Analysis
   → compares JD requirements vs resume capabilities
   → generates match score + prioritised gap list

Step 3 — Cover Letter Generation
   → 3 versions using candidate's actual experience

Step 4 — Interview Preparation
   → 10 questions with personalised suggested answers
```

Steps 1 runs JD and resume analysis simultaneously using `RunnableParallel` — cutting processing time by ~50%.

## Project structure

```
job-application-assistant/
├── app.py              # Streamlit UI
├── core/
│   ├── models.py       # Pydantic output models
│   ├── parser.py       # PDF and text parsing
│   └── chains.py       # LangChain LCEL pipeline
├── utils/
│   └── helpers.py      # Formatting and file saving
└── outputs/            # Generated files (gitignored)
```

## Run locally

**1. Clone the repo**
```bash
git clone https://github.com/thammi21/resumatch-ai
cd job-application-assistant
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your API key**

Get a free Groq API key at [console.groq.com](https://console.groq.com)

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```

**5. Run**
```bash
streamlit run app.py
```

## Key concepts demonstrated

- **LCEL pipe syntax** — `prompt | llm | parser` chains
- **RunnableParallel** — JD and resume analysis run simultaneously
- **PydanticOutputParser** — structured validated output at every step
- **RunnableLambda** — Python functions as Runnable chain components
- **Context window management** — automatic truncation for long documents
- **Dual LLM instances** — `temperature=0` for analysis, `temperature=0.7` for creative writing

## What I learned building this

- How to design a multi-step LangChain pipeline where each step's output feeds the next
- Why parallel chains matter — analysis time dropped from ~12s to ~6s
- How `PydanticOutputParser` injects format instructions directly into prompts
- The difference between `RunnableLambda` (any Python function) and `RunnablePassthrough` (preserve inputs while adding new keys)
- How to use callbacks for real-time progress tracking in Streamlit
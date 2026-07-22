# core/chains.py
"""
LangChain pipeline for the Job Application Assistant.

Week 3 concepts used:
  - ChatPromptTemplate       : structured multi-role prompts
  - ChatGroq                 : LLM as a Runnable
  - PydanticOutputParser     : structured typed output
  - LCEL pipe operator |     : chaining components
  - RunnableParallel         : running chains simultaneously
  - RunnablePassthrough      : preserving input through chain steps
  - RunnableLambda           : wrapping Python functions as Runnables
  - Callbacks + verbose      : debugging chain execution
"""

import os
import json
import time
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)
from langchain_groq import ChatGroq
from langchain_core.globals import set_verbose

from core.models import (
    JDAnalysis,
    ResumeAnalysis,
    SkillGapReport,
    CoverLetter,
    InterviewPrep,
    PipelineResult,
)
from core.parser import truncate_if_needed

load_dotenv()


# ── LLM Setup ─────────────────────────────────────────────────────
# Created once, reused across all chains
# temperature=0 for structured output — we want consistency not creativity
# temperature=0.7 for cover letters — we want natural flowing text

llm_structured = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0,       # deterministic — for analysis + extraction
    max_tokens=2048,
)

llm_creative = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0.7,     # creative — for cover letters
    max_tokens=2048,
)


# ════════════════════════════════════════════════════════════════
# CHAIN 1: JD ANALYSIS CHAIN
# ════════════════════════════════════════════════════════════════

# Step 1: Create the output parser
# PydanticOutputParser generates format instructions automatically
# These instructions get injected into the prompt telling the LLM
# exactly what JSON structure to return
jd_parser = PydanticOutputParser(pydantic_object=JDAnalysis)

# Step 2: Create the prompt template
# {format_instructions} is auto-filled by the parser
# {jd} is filled by the user's job description
jd_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a precise job description analyst.
Extract structured information from job descriptions accurately.
Always respond with valid JSON matching the required format exactly.

{format_instructions}"""
    ),
    (
        "user",
        """Analyse this job description and extract all required information:

JOB DESCRIPTION:
{jd}

Extract every piece of information accurately.
If information is not available, use 'Not specified'."""
    )
]).partial(format_instructions=jd_parser.get_format_instructions())
# .partial() pre-fills format_instructions so we only need to pass {jd}
# at runtime — cleaner API


# Step 3: Build the chain using LCEL pipe operator
# prompt formats → llm generates → parser validates and returns JDAnalysis object
jd_analysis_chain = jd_prompt | llm_structured | jd_parser


# ════════════════════════════════════════════════════════════════
# CHAIN 2: RESUME ANALYSIS CHAIN
# ════════════════════════════════════════════════════════════════

resume_parser = PydanticOutputParser(pydantic_object=ResumeAnalysis)

resume_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a precise resume analyst.
Extract structured information from resumes accurately.
Always respond with valid JSON matching the required format exactly.

{format_instructions}"""
    ),
    (
        "user",
        """Analyse this resume and extract all information:

RESUME:
{resume}

Be specific and accurate. Extract real skills and achievements mentioned."""
    )
]).partial(format_instructions=resume_parser.get_format_instructions())

resume_analysis_chain = resume_prompt | llm_structured | resume_parser


# ════════════════════════════════════════════════════════════════
# CHAIN 3: PARALLEL ANALYSIS
# Runs JD chain and Resume chain SIMULTANEOUSLY
# Saves time — both run at the same time instead of sequentially
# ════════════════════════════════════════════════════════════════

parallel_analysis = RunnableParallel(
    jd_analysis     = jd_analysis_chain,
    resume_analysis = resume_analysis_chain,
)
# Input:  {"jd": "...", "resume": "..."}
# Output: {"jd_analysis": JDAnalysis, "resume_analysis": ResumeAnalysis}
# Both chains run at the same time — cuts waiting time in half


# ════════════════════════════════════════════════════════════════
# CHAIN 4: SKILL GAP CHAIN
# Takes output of parallel analysis and compares JD vs Resume
# ════════════════════════════════════════════════════════════════

gap_parser = PydanticOutputParser(pydantic_object=SkillGapReport)

gap_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a career gap analyst.
Compare job requirements against candidate qualifications precisely.
Provide honest, actionable gap analysis.
Always respond with valid JSON matching the required format exactly.

{format_instructions}"""
    ),
    (
        "user",
        """Compare this candidate's profile against the job requirements:

JOB REQUIREMENTS:
- Title: {job_title}
- Required Skills: {required_skills}
- Nice to Have: {nice_to_have_skills}
- Experience: {experience_required}
- Responsibilities: {responsibilities}

CANDIDATE PROFILE:
- Name: {candidate_name}
- Current Role: {current_role}
- Experience: {total_experience_years}
- Technical Skills: {technical_skills}
- Projects: {notable_projects}
- Achievements: {achievements}

Provide a thorough, honest analysis of fit and gaps."""
    )
]).partial(format_instructions=gap_parser.get_format_instructions())

# RunnableLambda to extract fields from parallel analysis output
# The parallel chain returns {"jd_analysis": JDAnalysis, "resume_analysis": ResumeAnalysis}
# We need to flatten this into individual fields for the gap prompt
def prepare_gap_inputs(parallel_output: dict) -> dict:
    """
    Extract fields from parallel analysis output for gap chain.
    Converts Pydantic objects to primitive types for prompt formatting.
    """
    jd  : JDAnalysis    = parallel_output["jd_analysis"]
    res : ResumeAnalysis = parallel_output["resume_analysis"]

    return {
        # From JD analysis
        "job_title"          : jd.job_title,
        "required_skills"    : ", ".join(jd.required_skills),
        "nice_to_have_skills": ", ".join(jd.nice_to_have_skills),
        "experience_required": jd.experience_required,
        "responsibilities"   : "\n".join(f"- {r}" for r in jd.responsibilities),

        # From Resume analysis
        "candidate_name"      : res.candidate_name,
        "current_role"        : res.current_role,
        "total_experience_years": res.total_experience_years,
        "technical_skills"    : ", ".join(res.technical_skills),
        "notable_projects"    : "\n".join(f"- {p}" for p in res.notable_projects),
        "achievements"        : "\n".join(f"- {a}" for a in res.achievements),
    }

gap_input_preparer = RunnableLambda(prepare_gap_inputs)

skill_gap_chain = gap_input_preparer | gap_prompt | llm_structured | gap_parser


# ════════════════════════════════════════════════════════════════
# CHAIN 5: COVER LETTER CHAIN
# ════════════════════════════════════════════════════════════════

cover_parser = PydanticOutputParser(pydantic_object=CoverLetter)

cover_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert cover letter writer.
Write compelling, personalised cover letters that highlight the candidate's
relevant experience for the specific role.
Never use generic phrases like "I am writing to express my interest".
Always be specific, confident, and concise.
Always respond with valid JSON matching the required format exactly.

{format_instructions}"""
    ),
    (
        "user",
        """Write 3 versions of a cover letter for this candidate:

ROLE: {job_title} at {company_name}
MATCH SCORE: {match_score}%

CANDIDATE STRENGTHS FOR THIS ROLE:
{candidate_strengths}

MATCHING SKILLS:
{matching_skills}

KEY ACHIEVEMENTS TO HIGHLIGHT:
{achievements}

CANDIDATE NAME: {candidate_name}
CURRENT ROLE: {current_role}

Write formal, conversational, and concise versions.
Make each version unique and compelling.
Reference specific skills and achievements."""
    )
]).partial(format_instructions=cover_parser.get_format_instructions())


def prepare_cover_inputs(inputs: dict) -> dict:
    """
    Prepare inputs for cover letter chain.
    inputs contains: gap_report, jd_analysis, resume_analysis
    """
    gap : SkillGapReport = inputs["gap_report"]
    jd  : JDAnalysis     = inputs["jd_analysis"]
    res : ResumeAnalysis  = inputs["resume_analysis"]

    return {
        "job_title"          : jd.job_title,
        "company_name"       : jd.company_name,
        "match_score"        : gap.match_score,
        "candidate_strengths": "\n".join(f"- {s}" for s in gap.candidate_strengths),
        "matching_skills"    : ", ".join(gap.matching_skills),
        "achievements"       : "\n".join(f"- {a}" for a in res.achievements),
        "candidate_name"     : res.candidate_name,
        "current_role"       : res.current_role,
    }

cover_input_preparer = RunnableLambda(prepare_cover_inputs)

cover_letter_chain = cover_input_preparer | cover_prompt | llm_creative | cover_parser


# ════════════════════════════════════════════════════════════════
# CHAIN 6: INTERVIEW PREP CHAIN
# ════════════════════════════════════════════════════════════════

interview_parser = PydanticOutputParser(pydantic_object=InterviewPrep)

interview_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert interview coach with deep knowledge of
tech company hiring processes in India and globally.
Generate realistic, role-specific interview questions and answers.
Always respond with valid JSON matching the required format exactly.

{format_instructions}"""
    ),
    (
        "user",
        """Generate comprehensive interview preparation for this candidate:

ROLE: {job_title} at {company_name}
ROLE LEVEL: {role_level}

REQUIRED SKILLS TO BE TESTED: {required_skills}
CANDIDATE'S TECHNICAL SKILLS: {technical_skills}
SKILL GAPS TO ADDRESS: {missing_skills}

CANDIDATE EXPERIENCE: {total_experience_years}
CANDIDATE PROJECTS: {notable_projects}

Generate 10 questions mixing technical, behavioural and culture-fit.
Suggested answers must reference the candidate's actual experience.
Red flags must be honest about potential interviewer concerns."""
    )
]).partial(format_instructions=interview_parser.get_format_instructions())


def prepare_interview_inputs(inputs: dict) -> dict:
    """Prepare inputs for interview prep chain."""
    gap : SkillGapReport = inputs["gap_report"]
    jd  : JDAnalysis     = inputs["jd_analysis"]
    res : ResumeAnalysis  = inputs["resume_analysis"]

    missing = [g.skill for g in gap.missing_critical_skills]

    return {
        "job_title"          : jd.job_title,
        "company_name"       : jd.company_name,
        "role_level"         : jd.role_level,
        "required_skills"    : ", ".join(jd.required_skills),
        "technical_skills"   : ", ".join(res.technical_skills),
        "missing_skills"     : ", ".join(missing) if missing else "None",
        "total_experience_years": res.total_experience_years,
        "notable_projects"   : "\n".join(f"- {p}" for p in res.notable_projects),
    }

interview_input_preparer = RunnableLambda(prepare_interview_inputs)

interview_prep_chain = (
    interview_input_preparer | interview_prompt | llm_structured | interview_parser
)


# ════════════════════════════════════════════════════════════════
# FULL PIPELINE ORCHESTRATION
# Ties all chains together in the correct order
# ════════════════════════════════════════════════════════════════

def run_pipeline(
    resume_text : str,
    jd_text     : str,
    verbose     : bool = False,
    on_step     : callable = None,
) -> PipelineResult:
    """
    Run the complete job application assistant pipeline.

    Args:
        resume_text : Parsed resume text (from parser.py)
        jd_text     : Parsed JD text (from parser.py)
        verbose     : If True, prints chain debug output
        on_step     : Optional callback function called after each step
                      Signature: on_step(step_name: str, status: str)
                      Used by Streamlit to update progress bar

    Returns:
        PipelineResult containing all 5 analysis objects

    Pipeline flow:
        Step 1: Parallel — JD analysis + Resume analysis (simultaneously)
        Step 2: Sequential — Skill gap analysis (needs step 1 output)
        Step 3: Sequential — Cover letter (needs step 2 output)
        Step 4: Sequential — Interview prep (needs step 1 + 2 output)

    Why not make steps 3+4 parallel?
        Cover letter and interview prep both need gap_report.
        gap_report is only available after step 2.
        So 3+4 must come after 2 — but could run parallel with each other.
        For simplicity and rate limit safety we run them sequentially.
        In production you'd run 3+4 in parallel too.
    """
    if verbose:
        set_verbose(True)

    def notify(step: str, status: str):
        """Notify progress — works with or without Streamlit callback."""
        print(f"  [{status}] {step}")
        if on_step:
            on_step(step, status)

    try:
        # ── Step 1: Parallel Analysis ─────────────────────────────
        notify("Analysing JD and Resume simultaneously", "running")
        start = time.time()

        parallel_result = parallel_analysis.invoke({
            "jd"    : truncate_if_needed(jd_text, max_tokens=2500),
            "resume": truncate_if_needed(resume_text, max_tokens=2500),
        })

        jd_analysis     = parallel_result["jd_analysis"]
        resume_analysis = parallel_result["resume_analysis"]

        notify(
            f"Analysis complete — {jd_analysis.job_title} / "
            f"{resume_analysis.candidate_name}",
            "done"
        )
        print(f"     Step 1 took {time.time() - start:.1f}s")

        # ── Step 2: Skill Gap Analysis ────────────────────────────
        notify("Calculating skill gaps", "running")
        start = time.time()

        gap_report = skill_gap_chain.invoke(parallel_result)

        notify(
            f"Gap analysis complete — {gap_report.match_score}% match",
            "done"
        )
        print(f"     Step 2 took {time.time() - start:.1f}s")

        # ── Steps 3+4 share the same input dict ──────────────────
        combined_input = {
            "gap_report"     : gap_report,
            "jd_analysis"    : jd_analysis,
            "resume_analysis": resume_analysis,
        }

        # ── Step 3: Cover Letter ──────────────────────────────────
        notify("Generating cover letters", "running")
        start = time.time()

        cover_letter = cover_letter_chain.invoke(combined_input)

        notify("Cover letters generated — 3 versions", "done")
        print(f"     Step 3 took {time.time() - start:.1f}s")

        # ── Step 4: Interview Prep ────────────────────────────────
        notify("Generating interview preparation", "running")
        start = time.time()

        interview_prep = interview_prep_chain.invoke(combined_input)

        notify(
            f"Interview prep complete — "
            f"{len(interview_prep.likely_questions)} questions",
            "done"
        )
        print(f"     Step 4 took {time.time() - start:.1f}s")

        # ── Assemble final result ─────────────────────────────────
        return PipelineResult(
            jd_analysis     = jd_analysis,
            resume_analysis = resume_analysis,
            skill_gap       = gap_report,
            cover_letter    = cover_letter,
            interview_prep  = interview_prep,
        )

    except Exception as e:
        if verbose:
            set_verbose(False)
        raise RuntimeError(f"Pipeline failed: {str(e)}") from e

    finally:
        if verbose:
            set_verbose(False)
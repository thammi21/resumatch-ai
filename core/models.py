# core/models.py
"""
Pydantic output models for the Job Application Assistant pipeline.

Each model represents the structured output of one chain step.
LangChain's PydanticOutputParser uses these to:
  1. Generate format instructions injected into the prompt
  2. Validate and parse the LLM's JSON response
  3. Give you a typed Python object instead of a raw dict

Week 3 concept: Structured output with Pydantic
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# ── Model 1: JD Analysis ─────────────────────────────────────────
class JDAnalysis(BaseModel):
    """
    Structured breakdown of a job description.
    Output of jd_analysis_chain.
    """

    job_title: str = Field(
        description="The exact job title from the JD"
    )
    company_name: str = Field(
        description="Company name if mentioned, else 'Not specified'"
    )
    required_skills: List[str] = Field(
        description="Must-have technical skills explicitly required in the JD"
    )
    nice_to_have_skills: List[str] = Field(
        description="Optional or preferred skills mentioned in the JD"
    )
    responsibilities: List[str] = Field(
        description="Top 5 key responsibilities of the role"
    )
    experience_required: str = Field(
        description="Years of experience required, e.g. '2-4 years'"
    )
    education_required: str = Field(
        description="Education qualification required, e.g. 'B.Tech CS or equivalent'"
    )
    company_culture_signals: List[str] = Field(
        description="Keywords revealing company culture e.g. 'fast-paced', 'remote-first'"
    )
    role_level: str = Field(
        description="Seniority level: Junior / Mid / Senior / Lead"
    )
    location: str = Field(
        description="Job location or 'Remote' if remote"
    )


# ── Model 2: Resume Analysis ──────────────────────────────────────
class ResumeAnalysis(BaseModel):
    """
    Structured breakdown of a candidate's resume.
    Output of resume_analysis_chain.
    """

    candidate_name: str = Field(
        description="Full name of the candidate"
    )
    current_role: str = Field(
        description="Current or most recent job title"
    )
    total_experience_years: str = Field(
        description="Total years of professional experience e.g. '3 years'"
    )
    technical_skills: List[str] = Field(
        description="All technical skills mentioned in the resume"
    )
    soft_skills: List[str] = Field(
        description="Soft skills mentioned or implied in the resume"
    )
    education: str = Field(
        description="Highest education qualification and institution"
    )
    notable_projects: List[str] = Field(
        description="Top 3 projects with brief one-line descriptions"
    )
    achievements: List[str] = Field(
        description="Quantified achievements e.g. 'Reduced API latency by 40%'"
    )
    certifications: List[str] = Field(
        description="Certifications listed, empty list if none"
    )
    career_summary: str = Field(
        description="2-sentence summary of candidate's profile"
    )


# ── Model 3: Skill Gap Report ─────────────────────────────────────
class SkillGap(BaseModel):
    """Single skill gap entry with context."""

    skill: str = Field(
        description="The missing skill name"
    )
    importance: str = Field(
        description="How critical this skill is: Critical / Important / Nice-to-have"
    )
    learning_time: str = Field(
        description="Estimated time to learn e.g. '2-4 weeks'"
    )
    resource_suggestion: str = Field(
        description="One specific resource to learn this skill"
    )


class SkillGapReport(BaseModel):
    """
    Comparison between JD requirements and resume capabilities.
    Output of skill_gap_chain.
    Uses both JDAnalysis and ResumeAnalysis as input.
    """

    match_score: int = Field(
        description="Overall match percentage between resume and JD, 0-100"
    )
    match_summary: str = Field(
        description="2-sentence summary of how well the candidate fits the role"
    )
    matching_skills: List[str] = Field(
        description="Skills that appear in both JD requirements and resume"
    )
    missing_critical_skills: List[SkillGap] = Field(
        description="Required skills completely absent from resume — critical gaps"
    )
    missing_optional_skills: List[SkillGap] = Field(
        description="Nice-to-have skills absent from resume"
    )
    candidate_strengths: List[str] = Field(
        description="Resume skills that are particularly strong for this role"
    )
    quick_wins: List[str] = Field(
        description="Skills candidate almost has — small effort to close gap"
    )
    overall_recommendation: str = Field(
        description="Should apply / Apply with upskilling / Not a good fit + reasoning"
    )


# ── Model 4: Cover Letter ─────────────────────────────────────────
class CoverLetterVersion(BaseModel):
    """Single version of a cover letter."""

    style: str = Field(
        description="Style of this version: Formal / Conversational / Concise"
    )
    subject_line: str = Field(
        description="Email subject line for this cover letter"
    )
    content: str = Field(
        description="Full cover letter text, 3-4 paragraphs"
    )
    word_count: int = Field(
        description="Approximate word count of the content"
    )


class CoverLetter(BaseModel):
    """
    Three versions of a tailored cover letter.
    Output of cover_letter_chain.
    """

    formal: CoverLetterVersion = Field(
        description="Professional formal cover letter"
    )
    conversational: CoverLetterVersion = Field(
        description="Friendly conversational cover letter"
    )
    concise: CoverLetterVersion = Field(
        description="Short punchy cover letter under 150 words"
    )
    key_points_used: List[str] = Field(
        description="Resume highlights used to personalise the letters"
    )


# ── Model 5: Interview Preparation ───────────────────────────────
class InterviewQuestion(BaseModel):
    """Single interview question with suggested answer."""

    question: str = Field(
        description="The interview question"
    )
    category: str = Field(
        description="Question type: Technical / Behavioural / Situational / Culture-fit"
    )
    difficulty: str = Field(
        description="Difficulty level: Easy / Medium / Hard"
    )
    suggested_answer: str = Field(
        description="Personalised suggested answer based on candidate's resume"
    )
    answer_framework: str = Field(
        description="Framework to use: STAR / Concept explanation / Problem-solution"
    )


class InterviewPrep(BaseModel):
    """
    Complete interview preparation package.
    Output of interview_prep_chain.
    """

    likely_questions: List[InterviewQuestion] = Field(
        description="10 most likely interview questions for this specific role"
    )
    technical_topics_to_revise: List[str] = Field(
        description="Technical concepts to brush up on before the interview"
    )
    questions_to_ask_interviewer: List[str] = Field(
        description="5 smart questions candidate should ask the interviewer"
    )
    red_flags_to_address: List[str] = Field(
        description="Potential concerns interviewer might raise about candidate"
    )
    preparation_checklist: List[str] = Field(
        description="Step by step checklist to prepare for this interview"
    )


# ── Model 6: Full Pipeline Result ────────────────────────────────
class PipelineResult(BaseModel):
    """
    Complete output of the full pipeline.
    Wraps all individual results into one object.
    Passed from chains.py to app.py.
    """

    jd_analysis    : JDAnalysis
    resume_analysis: ResumeAnalysis
    skill_gap      : SkillGapReport
    cover_letter   : CoverLetter
    interview_prep : InterviewPrep
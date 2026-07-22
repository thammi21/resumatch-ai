# utils/helpers.py
"""
Helper functions for the Job Application Assistant.

Handles:
  - Formatting pipeline results for Streamlit display
  - Saving outputs to disk (JSON + text files)
  - Creating downloadable ZIP bundle
  - Display utility functions

Week 1 concepts revisited: File I/O, JSON, pathlib, datetime
New: zipfile module for bundling outputs
"""

import json
import zipfile
from pathlib import Path
from datetime import datetime

from core.models import (
    PipelineResult,
    SkillGapReport,
    CoverLetter,
    InterviewPrep,
    JDAnalysis,
    ResumeAnalysis,
)


# ── Output directory ──────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Timestamp helper ──────────────────────────────────────────────
def get_timestamp() -> str:
    """Return current timestamp string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ── Match score colour ────────────────────────────────────────────
def get_match_colour(score: int) -> str:
    """
    Return Streamlit colour string based on match score.
    Used to colour the match percentage display.
    """
    if score >= 75:
        return "green"
    elif score >= 50:
        return "orange"
    else:
        return "red"


def get_match_label(score: int) -> str:
    """Return label based on match score."""
    if score >= 75:
        return "Strong Match ✅"
    elif score >= 50:
        return "Moderate Match ⚠️"
    else:
        return "Weak Match ❌"


# ── Format skill gap for display ──────────────────────────────────
def format_skill_gap_md(gap: SkillGapReport) -> str:
    """
    Format SkillGapReport as Markdown string for Streamlit display.

    Returns a complete markdown document showing:
    - Match score and summary
    - Matching skills
    - Missing critical skills with learning resources
    - Quick wins
    - Overall recommendation
    """
    lines = []

    # Match score header
    label = get_match_label(gap.match_score)
    lines.append(f"## {label} — {gap.match_score}% Match\n")
    lines.append(f"{gap.match_summary}\n")

    # Matching skills
    if gap.matching_skills:
        lines.append("### ✅ Matching Skills")
        skills_row = " · ".join(
            [f"`{s}`" for s in gap.matching_skills]
        )
        lines.append(skills_row + "\n")

    # Missing critical skills
    if gap.missing_critical_skills:
        lines.append("### ❌ Missing Critical Skills")
        for skill_gap in gap.missing_critical_skills:
            lines.append(
                f"**{skill_gap.skill}** — "
                f"*{skill_gap.importance}* · "
                f"~{skill_gap.learning_time} to learn"
            )
            lines.append(
                f"  → 📚 {skill_gap.resource_suggestion}"
            )
        lines.append("")

    # Missing optional skills
    if gap.missing_optional_skills:
        lines.append("### 🟡 Nice-to-Have Gaps")
        for skill_gap in gap.missing_optional_skills:
            lines.append(
                f"- **{skill_gap.skill}** "
                f"(~{skill_gap.learning_time})"
            )
        lines.append("")

    # Strengths
    if gap.candidate_strengths:
        lines.append("### 💪 Your Strengths for This Role")
        for strength in gap.candidate_strengths:
            lines.append(f"- {strength}")
        lines.append("")

    # Quick wins
    if gap.quick_wins:
        lines.append("### ⚡ Quick Wins")
        for win in gap.quick_wins:
            lines.append(f"- {win}")
        lines.append("")

    # Recommendation
    lines.append("### 📋 Recommendation")
    lines.append(f"> {gap.overall_recommendation}")

    return "\n".join(lines)


# ── Format cover letter for display ──────────────────────────────
def format_cover_letter_md(cover: CoverLetter, style: str = "formal") -> str:
    """
    Format a single cover letter version as Markdown.

    Args:
        cover: CoverLetter Pydantic object
        style: "formal" | "conversational" | "concise"

    Returns:
        Markdown formatted cover letter
    """
    version_map = {
        "formal"         : cover.formal,
        "conversational" : cover.conversational,
        "concise"        : cover.concise,
    }

    version = version_map.get(style, cover.formal)

    lines = [
        f"**Subject:** {version.subject_line}\n",
        "---",
        version.content,
        "---",
        f"*~{version.word_count} words · {version.style} style*"
    ]

    return "\n\n".join(lines)


# ── Format interview prep for display ────────────────────────────
def format_interview_prep_md(prep: InterviewPrep) -> str:
    """
    Format InterviewPrep as Markdown for Streamlit display.
    Groups questions by category for easier reading.
    """
    lines = []

    # Group questions by category
    categories = {}
    for q in prep.likely_questions:
        cat = q.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(q)

    # Category icons
    icons = {
        "Technical"   : "💻",
        "Behavioural" : "🧠",
        "Situational" : "📋",
        "Culture-fit" : "🤝",
    }

    for category, questions in categories.items():
        icon = icons.get(category, "❓")
        lines.append(f"### {icon} {category} Questions\n")

        for i, q in enumerate(questions, 1):
            difficulty_emoji = {
                "Easy"  : "🟢",
                "Medium": "🟡",
                "Hard"  : "🔴"
            }.get(q.difficulty, "⚪")

            lines.append(
                f"**Q{i}. {q.question}** "
                f"{difficulty_emoji} *{q.difficulty}*"
            )
            lines.append(
                f"\n💡 **Suggested Answer:**\n{q.suggested_answer}"
            )
            lines.append(
                f"\n🎯 **Framework:** {q.answer_framework}\n"
            )
            lines.append("---")

    # Topics to revise
    if prep.technical_topics_to_revise:
        lines.append("### 📚 Topics to Revise Before Interview")
        for topic in prep.technical_topics_to_revise:
            lines.append(f"- {topic}")
        lines.append("")

    # Questions to ask
    if prep.questions_to_ask_interviewer:
        lines.append("### 🙋 Questions to Ask the Interviewer")
        for q in prep.questions_to_ask_interviewer:
            lines.append(f"- {q}")
        lines.append("")

    # Red flags
    if prep.red_flags_to_address:
        lines.append("### ⚠️ Potential Concerns to Address")
        for flag in prep.red_flags_to_address:
            lines.append(f"- {flag}")
        lines.append("")

    # Prep checklist
    if prep.preparation_checklist:
        lines.append("### ✅ Preparation Checklist")
        for item in prep.preparation_checklist:
            lines.append(f"- [ ] {item}")

    return "\n".join(lines)


# ── Format JD analysis for display ───────────────────────────────
def format_jd_analysis_md(jd: JDAnalysis) -> str:
    """Format JDAnalysis as Markdown."""
    lines = [
        f"## {jd.job_title}",
        f"**Company:** {jd.company_name} · "
        f"**Level:** {jd.role_level} · "
        f"**Location:** {jd.location}\n",

        "### 📋 Required Skills",
        " · ".join([f"`{s}`" for s in jd.required_skills]) + "\n",

        "### 🌟 Nice to Have",
        " · ".join([f"`{s}`" for s in jd.nice_to_have_skills]) + "\n",

        "### 📌 Key Responsibilities",
    ]
    for r in jd.responsibilities:
        lines.append(f"- {r}")

    lines += [
        f"\n**Experience Required:** {jd.experience_required}",
        f"**Education:** {jd.education_required}",
    ]

    if jd.company_culture_signals:
        lines.append(
            "\n**Culture Signals:** " +
            " · ".join([f"*{c}*" for c in jd.company_culture_signals])
        )

    return "\n".join(lines)


# ── Format resume analysis for display ───────────────────────────
def format_resume_analysis_md(res: ResumeAnalysis) -> str:
    """Format ResumeAnalysis as Markdown."""
    lines = [
        f"## {res.candidate_name}",
        f"**Current Role:** {res.current_role} · "
        f"**Experience:** {res.total_experience_years}\n",

        f"*{res.career_summary}*\n",

        "### 🛠️ Technical Skills",
        " · ".join([f"`{s}`" for s in res.technical_skills]) + "\n",
    ]

    if res.soft_skills:
        lines += [
            "### 🤝 Soft Skills",
            " · ".join(res.soft_skills) + "\n",
        ]

    if res.notable_projects:
        lines.append("### 🚀 Notable Projects")
        for project in res.notable_projects:
            lines.append(f"- {project}")
        lines.append("")

    if res.achievements:
        lines.append("### 🏆 Key Achievements")
        for achievement in res.achievements:
            lines.append(f"- {achievement}")
        lines.append("")

    lines += [
        f"**Education:** {res.education}",
    ]

    if res.certifications:
        lines.append(
            "**Certifications:** " + ", ".join(res.certifications)
        )

    return "\n".join(lines)


# ── Save outputs to disk ──────────────────────────────────────────
def save_outputs(result: PipelineResult) -> dict:
    """
    Save all pipeline outputs to disk.
    Creates a timestamped folder with all results.

    Returns dict of saved file paths.
    """
    timestamp = get_timestamp()
    session_dir = OUTPUT_DIR / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # ── Save full JSON result ──
    json_path = session_dir / "full_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            result.model_dump(),
            f,
            indent=2,
            ensure_ascii=False
        )
    saved_files["json"] = json_path

    # ── Save skill gap report ──
    gap_path = session_dir / "skill_gap_report.md"
    with open(gap_path, "w", encoding="utf-8") as f:
        f.write("# Skill Gap Report\n\n")
        f.write(format_skill_gap_md(result.skill_gap))
    saved_files["gap_report"] = gap_path

    # ── Save cover letters ──
    for style in ["formal", "conversational", "concise"]:
        cl_path = session_dir / f"cover_letter_{style}.md"
        with open(cl_path, "w", encoding="utf-8") as f:
            f.write(f"# Cover Letter — {style.title()}\n\n")
            f.write(format_cover_letter_md(result.cover_letter, style))
        saved_files[f"cover_{style}"] = cl_path

    # ── Save interview prep ──
    interview_path = session_dir / "interview_prep.md"
    with open(interview_path, "w", encoding="utf-8") as f:
        f.write("# Interview Preparation\n\n")
        f.write(format_interview_prep_md(result.interview_prep))
    saved_files["interview_prep"] = interview_path

    print(f"  Outputs saved to: {session_dir}")
    return saved_files


# ── Create downloadable ZIP ───────────────────────────────────────
def create_zip_bundle(result: PipelineResult) -> bytes:
    """
    Create a ZIP file containing all outputs as bytes.
    Used by Streamlit's st.download_button.

    Returns ZIP file as bytes — no file written to disk.
    """
    import io

    # Write ZIP to memory buffer — not to disk
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:

        # Full JSON
        zf.writestr(
            "full_result.json",
            json.dumps(result.model_dump(), indent=2, ensure_ascii=False)
        )

        # Skill gap report
        zf.writestr(
            "skill_gap_report.md",
            "# Skill Gap Report\n\n" + format_skill_gap_md(result.skill_gap)
        )

        # All 3 cover letters
        for style in ["formal", "conversational", "concise"]:
            zf.writestr(
                f"cover_letter_{style}.md",
                f"# Cover Letter — {style.title()}\n\n" +
                format_cover_letter_md(result.cover_letter, style)
            )

        # Interview prep
        zf.writestr(
            "interview_prep.md",
            "# Interview Preparation\n\n" +
            format_interview_prep_md(result.interview_prep)
        )

    # Return bytes — Streamlit download_button accepts bytes directly
    buffer.seek(0)
    return buffer.read()


# ── Summary stats for display ─────────────────────────────────────
def get_summary_stats(result: PipelineResult) -> dict:
    """
    Extract key numbers for the summary metrics display.
    Used in Streamlit's st.metric() calls.
    """
    return {
        "match_score"      : f"{result.skill_gap.match_score}%",
        "matching_skills"  : len(result.skill_gap.matching_skills),
        "missing_critical" : len(result.skill_gap.missing_critical_skills),
        "interview_questions": len(result.interview_prep.likely_questions),
        "cover_letter_versions": 3,
        "recommendation"   : result.skill_gap.overall_recommendation[:60] + "...",
    }
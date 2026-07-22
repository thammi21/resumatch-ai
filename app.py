# app.py
"""
Job Application Assistant — Streamlit UI

Connects all modules:
  core/parser.py   → PDF text extraction
  core/chains.py   → LangChain pipeline
  utils/helpers.py → formatting + ZIP download

UI Structure:
  Sidebar  → inputs (resume upload + JD input)
  Main     → 5 tabs (overview, JD, resume, cover letter, interview)
"""

import streamlit as st
import time
from core.parser import parse_resume, parse_jd, validate_inputs
from core.chains import run_pipeline
from utils.helpers import (
    format_skill_gap_md,
    format_cover_letter_md,
    format_interview_prep_md,
    format_jd_analysis_md,
    format_resume_analysis_md,
    get_match_colour,
    get_match_label,
    get_summary_stats,
    create_zip_bundle,
    save_outputs,
)

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Job Application Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark background */
    .stApp { background-color: #0f1117; }

    /* Hide default Streamlit chrome */
    #MainMenu  { visibility: hidden; }
    header     { visibility: hidden; }
    footer     { visibility: hidden; }

    /* Match score badge */
    .match-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 20px;
        font-size: 28px;
        font-weight: bold;
        margin: 10px 0;
    }
    .match-green  { background: #052e16; color: #4ade80; border: 1px solid #4ade80; }
    .match-orange { background: #431407; color: #fb923c; border: 1px solid #fb923c; }
    .match-red    { background: #3f0000; color: #f87171; border: 1px solid #f87171; }

    /* Step progress */
    .step-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
        border-left: 3px solid #334155;
        font-size: 14px;
        color: #94a3b8;
    }
    .step-running {
        border-left-color: #3b82f6;
        color: #93c5fd;
    }
    .step-done {
        border-left-color: #10b981;
        color: #6ee7b7;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #1e293b;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background: #3b82f6 !important;
        color: white !important;
    }

    /* Cover letter box */
    .cover-letter-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px 24px;
        font-family: Georgia, serif;
        line-height: 1.8;
        color: #e2e8f0;
        white-space: pre-wrap;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px;
    }

    /* Skill tags */
    .skill-tag {
        display: inline-block;
        background: #1e3a5f;
        color: #93c5fd;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 13px;
        margin: 3px;
    }
    .skill-tag-missing {
        background: #3f0000;
        color: #f87171;
    }

    /* Download button */
    .stDownloadButton button {
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────
def init_state():
    """Initialise all session state variables."""
    defaults = {
        "result"        : None,   # PipelineResult
        "resume_text"   : None,   # parsed resume string
        "jd_text"       : None,   # parsed JD string
        "pipeline_ran"  : False,  # whether pipeline has completed
        "step_statuses" : {},     # step name → status for progress display
        "error"         : None,   # error message if pipeline fails
        "run_time"      : None,   # total pipeline run time
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

init_state()


# ── Sidebar ───────────────────────────────────────────────────────
def render_sidebar():
    """Render input sidebar — resume upload + JD input."""

    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding:10px 0 20px;'>
            <h2 style='color:#3b82f6; margin:0;'>💼 Job Application</h2>
            <h2 style='color:#3b82f6; margin:0;'>Assistant</h2>
            <p style='color:#475569; font-size:12px; margin-top:6px;'>
                Powered by Groq · Llama 3.3
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Resume Upload ──
        st.markdown("### 📄 Your Resume")
        resume_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            help="Upload your resume as a PDF file",
            key="resume_uploader"
        )

        if resume_file:
            try:
                resume_text = parse_resume(resume_file)
                st.session_state.resume_text = resume_text
                word_count = len(resume_text.split())
                st.success(f"✓ Resume loaded — {word_count} words")
            except Exception as e:
                st.error(f"Resume error: {str(e)}")
                st.session_state.resume_text = None

        st.divider()

        # ── Job Description Input ──
        st.markdown("### 📋 Job Description")

        jd_input_type = st.radio(
            "Input method",
            ["Paste text", "Upload PDF"],
            horizontal=True,
            label_visibility="collapsed"
        )

        if jd_input_type == "Paste text":
            jd_text_input = st.text_area(
                "Paste the job description here",
                height=200,
                placeholder="Paste the full job description...",
                label_visibility="collapsed"
            )
            if jd_text_input and len(jd_text_input.strip()) > 30:
                try:
                    jd_text = parse_jd(jd_text_input, input_type="text")
                    st.session_state.jd_text = jd_text
                    word_count = len(jd_text.split())
                    st.success(f"✓ JD loaded — {word_count} words")
                except Exception as e:
                    st.error(str(e))
                    st.session_state.jd_text = None

        else:
            jd_file = st.file_uploader(
                "Upload JD PDF",
                type=["pdf"],
                key="jd_uploader"
            )
            if jd_file:
                try:
                    jd_text = parse_jd(jd_file, input_type="pdf")
                    st.session_state.jd_text = jd_text
                    word_count = len(jd_text.split())
                    st.success(f"✓ JD loaded — {word_count} words")
                except Exception as e:
                    st.error(str(e))
                    st.session_state.jd_text = None

        st.divider()

        # ── Analyse Button ──
        both_ready = (
            st.session_state.resume_text is not None and
            st.session_state.jd_text is not None
        )

        analyse_btn = st.button(
            "🚀 Analyse Application",
            use_container_width=True,
            disabled=not both_ready,
            type="primary"
        )

        if not both_ready:
            st.caption("Upload resume and add JD to enable analysis")

        # ── Reset Button ──
        if st.session_state.pipeline_ran:
            if st.button("🔄 Start New Analysis", use_container_width=True):
                for key in ["result", "resume_text", "jd_text",
                            "pipeline_ran", "step_statuses",
                            "error", "run_time"]:
                    st.session_state[key] = None \
                        if key != "pipeline_ran" else False
                    if key == "step_statuses":
                        st.session_state[key] = {}
                st.rerun()

        return analyse_btn


# ── Pipeline progress display ─────────────────────────────────────
def render_progress():
    """
    Show pipeline steps with live status updates.
    Called during pipeline execution.
    """
    steps = [
        "Analysing JD and Resume simultaneously",
        "Calculating skill gaps",
        "Generating cover letters",
        "Generating interview preparation",
    ]

    st.markdown("### ⚡ Running Analysis Pipeline")
    st.caption(
        "Step 1 runs JD + Resume analysis in parallel — "
        "saving ~50% of processing time"
    )

    placeholders = {}
    for step in steps:
        placeholders[step] = st.empty()
        # Render initial pending state
        placeholders[step].markdown(
            f"<div class='step-card'>⏳ {step}</div>",
            unsafe_allow_html=True
        )

    return placeholders


# ── Main results area ─────────────────────────────────────────────
def render_results(result):
    """
    Render all pipeline results in a tabbed interface.
    5 tabs: Overview, JD Analysis, Resume Analysis,
            Cover Letters, Interview Prep
    """

    stats = get_summary_stats(result)
    score = result.skill_gap.match_score
    colour = get_match_colour(score)
    label = get_match_label(score)

    # ── Header with match score ──
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(
            f"<h2 style='color:#e2e8f0; margin:0;'>"
            f"{result.jd_analysis.job_title}</h2>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='color:#64748b;'>"
            f"{result.jd_analysis.company_name} · "
            f"{result.jd_analysis.location} · "
            f"{result.jd_analysis.role_level}</p>",
            unsafe_allow_html=True
        )

    with col2:
        badge_class = f"match-{colour}"
        st.markdown(
            f"<div class='match-badge {badge_class}'>"
            f"{score}% Match</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='color:#64748b; font-size:13px;'>{label}</p>",
            unsafe_allow_html=True
        )

    st.divider()

    # ── Summary metrics ──
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("Match Score", stats["match_score"])
    with c2:
        st.metric("Matching Skills", stats["matching_skills"])
    with c3:
        st.metric("Critical Gaps", stats["missing_critical"])
    with c4:
        st.metric("Interview Questions", stats["interview_questions"])
    with c5:
        st.metric("Cover Letter Versions", stats["cover_letter_versions"])

    st.divider()

    # ── Download button ──
    zip_bytes = create_zip_bundle(result)
    st.download_button(
        label="📥 Download Full Package (ZIP)",
        data=zip_bytes,
        file_name=f"job_application_{result.jd_analysis.job_title.replace(' ', '_')}.zip",
        mime="application/zip",
        use_container_width=False,
    )

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Skill Gap",
        "📋 JD Analysis",
        "👤 Resume Analysis",
        "✉️ Cover Letters",
        "🎯 Interview Prep",
    ])

    # Tab 1 — Skill Gap
    with tab1:
        st.markdown(
            format_skill_gap_md(result.skill_gap)
        )

    # Tab 2 — JD Analysis
    with tab2:
        st.markdown(
            format_jd_analysis_md(result.jd_analysis)
        )

    # Tab 3 — Resume Analysis
    with tab3:
        st.markdown(
            format_resume_analysis_md(result.resume_analysis)
        )

    # Tab 4 — Cover Letters
    with tab4:
        st.markdown("### ✉️ Cover Letters — 3 Versions")
        st.caption(
            "Each version is tailored to your specific experience "
            "and the role requirements"
        )

        # Style selector
        style = st.radio(
            "Select version",
            ["formal", "conversational", "concise"],
            horizontal=True,
            format_func=lambda x: {
                "formal"        : "📝 Formal",
                "conversational": "💬 Conversational",
                "concise"       : "⚡ Concise",
            }[x]
        )

        # Display selected version
        version_map = {
            "formal"        : result.cover_letter.formal,
            "conversational": result.cover_letter.conversational,
            "concise"       : result.cover_letter.concise,
        }
        version = version_map[style]

        st.markdown(
            f"**Subject:** `{version.subject_line}`"
        )
        st.markdown(
            f"<div class='cover-letter-box'>{version.content}</div>",
            unsafe_allow_html=True
        )
        st.caption(f"~{version.word_count} words · {version.style} style")

        # Individual download
        st.download_button(
            label=f"📥 Download {style.title()} Cover Letter",
            data=version.content,
            file_name=f"cover_letter_{style}.txt",
            mime="text/plain",
        )

    # Tab 5 — Interview Prep
    with tab5:
        st.markdown(
            format_interview_prep_md(result.interview_prep)
        )


# ── Main app ──────────────────────────────────────────────────────
def main():

    # ── Render sidebar and get button state ──
    analyse_clicked = render_sidebar()

    # ── Run pipeline when button clicked ──
    if analyse_clicked:
        # Validate inputs
        is_valid, error_msg = validate_inputs(
            st.session_state.resume_text or "",
            st.session_state.jd_text or ""
        )

        if not is_valid:
            st.error(f"❌ {error_msg}")
            return

        # Clear previous results
        st.session_state.result = None
        st.session_state.error = None
        st.session_state.pipeline_ran = False

        # Show progress UI
        placeholders = render_progress()

        # Track steps for live UI updates
        step_map = {
            "Analysing JD and Resume simultaneously" : 0,
            "Calculating skill gaps"                 : 1,
            "Generating cover letters"               : 2,
            "Generating interview preparation"       : 3,
        }

        steps_list = list(step_map.keys())

        def on_step(step_name: str, status: str):
            """
            Callback passed to run_pipeline().
            Updates the progress display in real time.
            """
            # Find matching step
            matched = None
            for step in steps_list:
                if step_name[:20] in step or step[:20] in step_name:
                    matched = step
                    break

            if matched is None:
                return

            if status == "running":
                placeholders[matched].markdown(
                    f"<div class='step-card step-running'>"
                    f"⚡ {matched}</div>",
                    unsafe_allow_html=True
                )
            elif status == "done":
                placeholders[matched].markdown(
                    f"<div class='step-card step-done'>"
                    f"✅ {matched}</div>",
                    unsafe_allow_html=True
                )

        # Run the pipeline
        start_time = time.time()

        try:
            result = run_pipeline(
                resume_text = st.session_state.resume_text,
                jd_text     = st.session_state.jd_text,
                verbose     = False,
                on_step     = on_step,
            )

            st.session_state.result      = result
            st.session_state.pipeline_ran = True
            st.session_state.run_time    = time.time() - start_time

            # Save outputs to disk
            save_outputs(result)

            # Rerun to show results
            st.rerun()

        except Exception as e:
            st.session_state.error = str(e)
            st.error(f"❌ Pipeline failed: {str(e)}")
            st.info(
                "Common fixes:\n"
                "- Check your GROQ_API_KEY in .env\n"
                "- Try a shorter resume or JD\n"
                "- Wait 30 seconds if rate limited"
            )
            return

    # ── Show banner if nothing has run yet ──
    if not st.session_state.pipeline_ran:
        st.markdown("""
        <div style='text-align:center; padding:60px 20px;'>
            <h1 style='color:#3b82f6; font-size:48px;'>💼</h1>
            <h2 style='color:#e2e8f0;'>Job Application Assistant</h2>
            <p style='color:#64748b; font-size:16px; max-width:500px; margin:0 auto;'>
                Upload your resume and paste a job description.
                Get a complete application package in under 30 seconds.
            </p>
            <br>
            <div style='display:flex; justify-content:center; gap:30px;
                        flex-wrap:wrap; margin-top:20px;'>
                <div style='color:#94a3b8; text-align:center;'>
                    <div style='font-size:24px;'>📊</div>
                    <div style='font-size:13px;'>Skill Gap Analysis</div>
                </div>
                <div style='color:#94a3b8; text-align:center;'>
                    <div style='font-size:24px;'>✉️</div>
                    <div style='font-size:13px;'>3 Cover Letter Versions</div>
                </div>
                <div style='color:#94a3b8; text-align:center;'>
                    <div style='font-size:24px;'>🎯</div>
                    <div style='font-size:13px;'>Interview Preparation</div>
                </div>
                <div style='color:#94a3b8; text-align:center;'>
                    <div style='font-size:24px;'>📥</div>
                    <div style='font-size:13px;'>Download Full Package</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Show results if pipeline has run ──
    elif st.session_state.result is not None:
        if st.session_state.run_time:
            st.caption(
                f"✅ Analysis completed in "
                f"{st.session_state.run_time:.1f}s"
            )
        render_results(st.session_state.result)


if __name__ == "__main__":
    main()
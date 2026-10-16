
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder

from evaluator import generate_questions, evaluate_answer
from transcriber import transcribe_audio
from history import load_history, save_interview, delete_history

load_dotenv()

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Mock Interview Coach",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================
# CUSTOM CSS - READABLE LIGHT THEME
# ==================================================

st.markdown(
    """
    <style>
    /* Main application background and default text */
    .stApp {
        background-color: #f5f7fb !important;
        color: #1f2937 !important;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #f5f7fb !important;
    }

    [data-testid="stHeader"] {
        background-color: #f5f7fb !important;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Main page text */
    .main p,
    .main li,
    .main label,
    .main span,
    .main div {
        color: inherit;
    }

    .main h1,
    .main h2,
    .main h3,
    .main h4 {
        color: #172554 !important;
    }

    .main p,
    .main li {
        color: #1f2937;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #172554 !important;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    /* Metrics: prevent white text on a white background */
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #dbe3ef;
        border-radius: 12px;
        padding: 18px;
    }

    [data-testid="stMetric"] label,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] *,
    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] * {
        color: #172554 !important;
        -webkit-text-fill-color: #172554 !important;
        opacity: 1 !important;
    }

    [data-testid="stMetricValue"] {
        font-weight: 700 !important;
    }

    /* Inputs */
    .stTextArea textarea,
    .stTextInput input {
        background-color: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #cbd5e1 !important;
    }

    .stSelectbox label,
    .stTextInput label,
    .stTextArea label,
    .stSlider label,
    .stNumberInput label {
        color: #1f2937 !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 42px;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #dbe3ef !important;
        border-radius: 10px;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span {
        color: #172554 !important;
    }

    /* Dataframes and charts */
    [data-testid="stDataFrame"] {
        background-color: #ffffff !important;
    }

    [data-testid="stVegaLiteChart"] {
        background-color: #ffffff !important;
        border-radius: 10px;
        padding: 8px;
    }

    /* Progress bars */
    [data-testid="stProgressBar"] {
        background-color: #dbeafe;
    }

    /* Captions */
    .main .stCaption,
    .main [data-testid="stCaptionContainer"] {
        color: #475569 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# SESSION STATE
# ==================================================

DEFAULTS = {
    "page": "setup",
    "role": "Python Developer",
    "experience": "Fresher",
    "difficulty": "Medium",
    "question_type": "Mixed",
    "question_count": 5,
    "questions": [],
    "answers": {},
    "results": [],
    "interview_started": False,
    "interview_saved": False,
    "confirm_delete_history": False,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def normalize_score(value):
    """Convert a score to a 0-10 scale."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0

    if 10 < score <= 100:
        score = score / 10

    return round(max(0.0, min(10.0, score)), 1)


def get_question_text(item):
    if isinstance(item, dict):
        return str(item.get("question", "Question unavailable"))
    return str(item)


def clear_interview_widget_state():
    prefixes = (
        "answer_text_",
        "audio_bytes_",
        "pending_transcript_",
        "question_mic_",
        "transcribe_question_",
    )

    for key in list(st.session_state.keys()):
        if key.startswith(prefixes):
            del st.session_state[key]


def reset_interview():
    clear_interview_widget_state()

    st.session_state.questions = []
    st.session_state.answers = {}
    st.session_state.results = []
    st.session_state.interview_started = False
    st.session_state.interview_saved = False
    st.session_state.page = "setup"


def render_feedback(result):
    score = normalize_score(result.get("score", 0))
    st.metric("Answer Score", f"{score}/10")

    strengths = result.get("strengths", [])
    improvements = result.get("improvements", [])
    ideal_answer = result.get("ideal_answer", "")
    summary = result.get("summary", "")

    if isinstance(strengths, str):
        strengths = [strengths]

    if isinstance(improvements, str):
        improvements = [improvements]

    st.markdown("#### ✅ Strengths")

    if strengths:
        for item in strengths:
            st.markdown(f"- {item}")
    else:
        st.write("No strengths provided.")

    st.markdown("#### 🛠️ Areas to improve")

    if improvements:
        for item in improvements:
            st.markdown(f"- {item}")
    else:
        st.write("No improvement suggestions provided.")

    if summary:
        st.markdown("#### 📝 Overall feedback")
        st.write(summary)

    if ideal_answer:
        with st.expander("💡 View a sample ideal answer"):
            st.write(ideal_answer)


# ==================================================
# SIDEBAR
# ==================================================

def render_sidebar():
    with st.sidebar:
        st.title("🎯 Interview Coach")
        st.caption("Practice. Improve. Succeed.")
        st.divider()

        if st.button("🏠 New Interview", use_container_width=True):
            reset_interview()
            st.rerun()

        if st.button("📚 Interview History", use_container_width=True):
            st.session_state.page = "history"
            st.rerun()

        if st.button("ℹ️ About", use_container_width=True):
            st.session_state.page = "about"
            st.rerun()

        st.divider()
        st.markdown("### Current interview")
        st.write(f"**Role:** {st.session_state.role}")
        st.write(f"**Experience:** {st.session_state.experience}")
        st.write(f"**Difficulty:** {st.session_state.difficulty}")
        st.divider()
        st.caption("Powered by Groq AI")


# ==================================================
# SETUP PAGE
# ==================================================

def render_setup():
    st.title("🎯 AI Mock Interview Coach")
    st.write(
        "Practice for interviews using AI-generated questions, "
        "voice or text answers, and personalized feedback."
    )

    st.divider()
    st.subheader("⚙️ Configure your interview")

    with st.form("interview_setup_form"):
        role_option = st.selectbox(
            "Job role",
            [
                "Python Developer",
                "Data Scientist",
                "Data Analyst",
                "Machine Learning Engineer",
                "Frontend Developer",
                "Backend Developer",
                "Full Stack Developer",
                "Software Engineer",
                "Java Developer",
                "DevOps Engineer",
                "Product Manager",
                "Other",
            ],
        )

        custom_role = ""
        if role_option == "Other":
            custom_role = st.text_input("Enter your job role")

        experience = st.selectbox(
            "Experience level",
            ["Fresher", "0-2 years", "3-5 years", "6-10 years", "10+ years"],
        )

        difficulty = st.selectbox(
            "Question difficulty",
            ["Easy", "Medium", "Hard"],
            index=1,
        )

        question_type = st.selectbox(
            "Question type",
            ["Mixed", "Technical", "Behavioral", "Situational"],
        )

        question_count = st.slider(
            "Number of questions",
            min_value=3,
            max_value=10,
            value=5,
        )

        submitted = st.form_submit_button(
            "🚀 Generate Interview Questions",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        role = custom_role.strip() if role_option == "Other" else role_option

        if not role:
            st.error("Please enter a job role.")
            return

        try:
            with st.spinner("Generating interview questions..."):
                questions = generate_questions(
                    role=role,
                    experience=experience,
                    difficulty=difficulty,
                    question_count=question_count,
                    question_type=question_type,
                )

            if not questions:
                st.error("No questions were generated. Please try again.")
                return

            clear_interview_widget_state()

            st.session_state.role = role
            st.session_state.experience = experience
            st.session_state.difficulty = difficulty
            st.session_state.question_type = question_type
            st.session_state.question_count = question_count
            st.session_state.questions = questions
            st.session_state.answers = {}
            st.session_state.results = []
            st.session_state.interview_started = True
            st.session_state.interview_saved = False
            st.session_state.page = "interview"

            st.rerun()

        except Exception as error:
            st.error(f"Could not generate questions: {error}")


# ==================================================
# INTERVIEW PAGE
# ==================================================

def render_interview():
    st.title("🎙️ Mock Interview")
    st.write(
        "Record your answer or type it into the text box. "
        "Transcribed answers will be saved."
    )

    questions = st.session_state.get("questions", [])

    if not questions:
        st.error("No questions found. Please start a new interview.")
        if st.button("Back to Setup"):
            st.session_state.page = "setup"
            st.rerun()
        return

    for index, item in enumerate(questions):
        question = get_question_text(item)

        answer_key = f"answer_text_{index}"
        audio_key = f"audio_bytes_{index}"
        pending_key = f"pending_transcript_{index}"

        st.divider()
        st.subheader(f"Question {index + 1} of {len(questions)}")

        if isinstance(item, dict):
            category = item.get("category")
            difficulty = item.get("difficulty")

            if category or difficulty:
                st.caption(
                    f"Category: {category or 'General'} · "
                    f"Difficulty: {difficulty or st.session_state.difficulty}"
                )

        st.markdown(f"**{question}**")
        st.markdown("**🎤 Record your answer**")

        # Set pending transcript before constructing the text-area widget.
        if pending_key in st.session_state:
            st.session_state[answer_key] = st.session_state.pop(pending_key)

        if answer_key not in st.session_state:
            st.session_state[answer_key] = st.session_state.answers.get(
                str(index), ""
            )

        # A distinct widget key is used for each question.
        try:
            recording = mic_recorder(
                start_prompt="🔴 Start recording",
                stop_prompt="⏹️ Stop recording",
                just_once=True,
                use_container_width=True,
                format="webm",
                key=f"question_mic_{index}",
            )

            if recording and recording.get("bytes"):
                st.session_state[audio_key] = recording["bytes"]

        except Exception as error:
            st.warning(
                f"Microphone unavailable for question {index + 1}. "
                "You can still type your answer."
            )
            st.caption(str(error))

        if audio_key in st.session_state:
            st.audio(st.session_state[audio_key], format="audio/webm")

            if st.button(
                f"✨ Transcribe answer {index + 1}",
                key=f"transcribe_question_{index}",
                type="primary",
                use_container_width=True,
            ):
                try:
                    with st.spinner(
                        f"Transcribing answer {index + 1}..."
                    ):
                        transcript = transcribe_audio(
                            st.session_state[audio_key],
                            filename="answer.webm",
                        )

                    if transcript and transcript.strip():
                        transcript = transcript.strip()

                        st.session_state.answers[str(index)] = transcript
                        st.session_state[pending_key] = transcript
                        del st.session_state[audio_key]

                        st.success(
                            f"Answer {index + 1} transcribed successfully."
                        )
                        st.rerun()
                    else:
                        st.warning(
                            "No speech was detected. Please record again."
                        )

                except Exception as error:
                    st.error(f"Transcription failed: {error}")

        typed_answer = st.text_area(
            "Your answer",
            key=answer_key,
            height=160,
            placeholder="Type your answer here, or record it above...",
        )

        st.session_state.answers[str(index)] = typed_answer

        if typed_answer.strip():
            st.caption("✅ Answer saved")
        else:
            st.caption("Waiting for your answer")

    st.divider()

    answered = sum(
        bool(st.session_state.answers.get(str(i), "").strip())
        for i in range(len(questions))
    )

    st.progress(answered / len(questions))
    st.write(f"**{answered} of {len(questions)} questions answered**")

    if st.button(
        "📊 Evaluate My Answers",
        type="primary",
        use_container_width=True,
    ):
        unanswered = [
            i + 1
            for i in range(len(questions))
            if not st.session_state.answers.get(str(i), "").strip()
        ]

        if unanswered:
            st.warning(
                "Please answer questions: "
                + ", ".join(map(str, unanswered))
            )
        else:
            evaluate_all_answers()


# ==================================================
# EVALUATION
# ==================================================

def evaluate_all_answers():
    questions = st.session_state.get("questions", [])
    answers = st.session_state.get("answers", {})

    if not questions:
        st.error("There are no questions to evaluate.")
        return

    results = []
    progress = st.progress(0)
    status = st.empty()

    try:
        for index, item in enumerate(questions):
            question = get_question_text(item)
            answer = answers.get(str(index), "").strip()

            if not answer:
                progress.empty()
                status.empty()
                st.error(f"Question {index + 1} has no answer.")
                return

            status.write(
                f"Evaluating question {index + 1} of {len(questions)}..."
            )

            evaluation = evaluate_answer(
                question=question,
                answer=answer,
                role=st.session_state.role,
                experience=st.session_state.experience,
            )

            if not isinstance(evaluation, dict):
                raise ValueError(
                    f"Invalid evaluation returned for question {index + 1}."
                )

            evaluation["score"] = normalize_score(
                evaluation.get("score", 0)
            )
            evaluation["question"] = question
            evaluation["answer"] = answer

            results.append(evaluation)
            progress.progress((index + 1) / len(questions))

        # Save the results to session state before saving history.
        st.session_state.results = results
        st.session_state.page = "results"

        progress.empty()
        status.empty()

        if not st.session_state.get("interview_saved", False):
            average = sum(
                result["score"] for result in results
            ) / len(results)

            record = {
                "role": st.session_state.role,
                "experience": st.session_state.experience,
                "difficulty": st.session_state.difficulty,
                "question_type": st.session_state.question_type,
                "total_score": round(average, 1),
                "results": results,
            }

            try:
                save_interview(record)
                st.session_state.interview_saved = True
            except Exception as history_error:
                st.warning(
                    "Results are ready, but saving history failed: "
                    f"{history_error}"
                )

        st.rerun()

    except Exception as error:
        progress.empty()
        status.empty()

        st.error(f"Evaluation failed: {error}")
        st.info(
            "Check the Groq API key, internet connection, and evaluator "
            "configuration. Read the error above for details."
        )


# ==================================================
# RESULTS PAGE
# ==================================================

def render_results():
    st.title("📊 Interview Results")

    results = st.session_state.get("results", [])

    if not results:
        st.warning("No evaluation results are available.")
        if st.button("Return to Interview"):
            st.session_state.page = "interview"
            st.rerun()
        return

    scores = [
        normalize_score(result.get("score", 0))
        for result in results
    ]

    average_score = sum(scores) / len(scores)

    # Explicit dark text and white cards make scores readable.
    st.markdown(
        """
        <style>
        [data-testid="stMetric"] {
            background: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 12px !important;
            padding: 20px !important;
        }

        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] *,
        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] * {
            color: #172554 !important;
            -webkit-text-fill-color: #172554 !important;
            opacity: 1 !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Overall Score", f"{average_score:.1f}/10")

    with col2:
        st.metric("Questions Evaluated", len(results))

    with col3:
        st.metric(
            "Strong Answers",
            sum(score >= 7 for score in scores),
        )

    st.progress(average_score / 10)

    if average_score >= 8:
        st.success("Excellent work! Your answers were strong.")
    elif average_score >= 6:
        st.info("Good effort! Review your feedback to improve further.")
    else:
        st.warning(
            "Keep practicing and focus on the improvement suggestions."
        )

    # Score chart: do not pass unsupported y_min or y_max arguments.
    st.subheader("📈 Score by Question")

    chart_data = pd.DataFrame(
        {
            "Question": [f"Q{i + 1}" for i in range(len(results))],
            "Score": scores,
        }
    ).set_index("Question")

    st.bar_chart(chart_data, y="Score")

    # Also show scores in a table so values remain visible without the chart.
    score_table = pd.DataFrame(
        {
            "Question": [f"Question {i + 1}" for i in range(len(results))],
            "Score (/10)": scores,
        }
    )

    st.dataframe(
        score_table,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("📝 Detailed Feedback")

    for index, result in enumerate(results):
        score = normalize_score(result.get("score", 0))

        with st.expander(
            f"Question {index + 1} — Score: {score}/10",
            expanded=(index == 0),
        ):
            st.markdown("#### Question")
            st.write(result.get("question", "Question unavailable"))

            st.markdown("#### Your answer")
            st.write(result.get("answer", "No answer provided."))

            render_feedback(result)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "🔄 Start Another Interview",
            type="primary",
            use_container_width=True,
        ):
            reset_interview()
            st.rerun()

    with col2:
        if st.button(
            "📚 View Interview History",
            use_container_width=True,
        ):
            st.session_state.page = "history"
            st.rerun()


# ==================================================
# HISTORY PAGE
# ==================================================

def render_history():
    st.title("📚 Interview History")
    st.write("Review previously completed interviews.")

    try:
        history = load_history()
    except Exception as error:
        st.error(f"Could not load interview history: {error}")
        return

    if not history:
        st.info("No interview history found. Complete an interview first.")
    else:
        for record in reversed(history):
            role = record.get("role", "Unknown role")
            experience = record.get("experience", "Not specified")
            score = normalize_score(record.get("total_score", 0))
            results = record.get("results", [])

            with st.expander(
                f"{role} · {experience} · Score: {score}/10"
            ):
                st.metric("Overall Score", f"{score}/10")

                st.write(
                    f"**Difficulty:** {record.get('difficulty', 'N/A')}"
                )
                st.write(
                    f"**Question type:** {record.get('question_type', 'N/A')}"
                )

                if results:
                    rows = []

                    for index, result in enumerate(results):
                        rows.append(
                            {
                                "Question": f"Q{index + 1}",
                                "Score (/10)": normalize_score(
                                    result.get("score", 0)
                                ),
                                "Summary": result.get("summary", ""),
                            }
                        )

                    st.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True,
                        hide_index=True,
                    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🏠 Back to Setup", use_container_width=True):
            st.session_state.page = "setup"
            st.rerun()

    with col2:
        if st.button("🗑️ Delete All History", use_container_width=True):
            st.session_state.confirm_delete_history = True

    if st.session_state.get("confirm_delete_history", False):
        st.warning("Are you sure you want to delete all saved history?")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Yes, delete history", type="primary"):
                try:
                    delete_history()
                    st.session_state.confirm_delete_history = False
                    st.success("Interview history deleted.")
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not delete history: {error}")

        with col2:
            if st.button("Cancel"):
                st.session_state.confirm_delete_history = False
                st.rerun()


# ==================================================
# ABOUT PAGE
# ==================================================

def render_about():
    st.title("ℹ️ About AI Mock Interview Coach")

    st.markdown(
        """
        This application helps you prepare for interviews with AI-generated
        questions and personalized feedback.

        ### Features

        - AI-generated interview questions
        - Voice recording for each question
        - Speech-to-text transcription
        - Scores out of 10
        - Strengths, improvement suggestions, and sample ideal answers
        - Saved interview history

        ### How to use

        1. Configure and generate an interview.
        2. Type or record an answer for every question.
        3. Transcribe recordings and check the text boxes.
        4. Click **Evaluate My Answers**.
        5. Review your scores and feedback.
        """
    )

    if st.button("🏠 Back to Setup", type="primary"):
        st.session_state.page = "setup"
        st.rerun()


# ==================================================
# MAIN APPLICATION
# ==================================================

def main():
    render_sidebar()

    page = st.session_state.get("page", "setup")

    if page == "setup":
        render_setup()
    elif page == "interview":
        render_interview()
    elif page == "results":
        render_results()
    elif page == "history":
        render_history()
    elif page == "about":
        render_about()
    else:
        st.session_state.page = "setup"
        st.rerun()


if __name__ == "__main__":
    main()

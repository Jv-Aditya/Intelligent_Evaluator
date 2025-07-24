import streamlit as st
import time
import json
from Actions import *
from dotenv import load_dotenv
import os

# Load environment variables (for consistency if needed)
load_dotenv()
hf_token = os.getenv("hf_token")

# === UI Setup ===
st.set_page_config(page_title="Intelligent Evaluator (Manual)", layout="centered")
st.title("🧠 Intelligent Evaluator - Manual Flow")

# === Session State Initialization ===
if "topic" not in st.session_state:
    st.session_state.topic = ""
if "tags" not in st.session_state:
    st.session_state.tags = []
if "beliefs" not in st.session_state:
    st.session_state.beliefs = {}
if "question" not in st.session_state:
    st.session_state.question = {}
if "score" not in st.session_state:
    st.session_state.score = None
if "current_tag" not in st.session_state:
    st.session_state.current_tag = ""
if "step" not in st.session_state:
    st.session_state.step = "start"

# === Step 1: Enter Topic ===
if st.session_state.step == "start":
    topic = st.text_input("Enter topic to evaluate:", value="Python")
    if st.button("Generate Tags"):
        st.session_state.topic = topic
        try:
            result = generate_tags(topic)
            st.session_state.tags = result.get("tags", [])
            st.session_state.beliefs = result.get("beliefs", {})
            st.session_state.step = "choose_tag"
            st.success("Tags generated successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Error generating tags: {e}")

# === Step 2: Choose Tag and Question Type ===
elif st.session_state.step == "choose_tag":
    st.subheader("Step 2: Choose a Tag and Question Type")
    tag = st.selectbox("Choose a tag to test:", st.session_state.tags)
    q_type = st.selectbox("Choose question type:", ["MCQ", "ShortAnswer", "Coding"])
    difficulty = st.selectbox("Difficulty:", ["easy", "medium", "hard"])
    if st.button("Generate Question"):
        try:
            question = generate_question(tag=[tag], type=q_type, difficulty=difficulty)
            st.session_state.question = question
            st.session_state.current_tag = tag
            st.session_state.step = "show_question"
            st.rerun()
        except Exception as e:
            st.error(f"Error generating question: {e}")

# === Step 3: Show Question and Input Answer ===
elif st.session_state.step == "show_question":
    st.subheader("Step 3: Answer the Question")
    q = st.session_state.question
    print(q)
    st.markdown(f"**Question:** {q['question']}")
    q_type = q["type"]

    if q_type == "MCQ":
        user_answer = st.radio("Choose your answer:", q["options"])
    elif q_type == "ShortAnswer":
        user_answer = st.text_input("Enter your answer:")
    elif q_type == "Coding":
        user_answer = st.text_area("Write your code:", height=200)
    else:
        st.error("Unknown question type.")
        user_answer = None

    if st.button("Submit Answer"):
        score = 0
        try:
            if q_type == "MCQ":
                score = evaluate_mcq([user_answer], q["correct_answer"])
            elif q_type == "ShortAnswer":
                score = float(evaluate_short_answer(user_answer, q["correct_answer"]))
            elif q_type == "Coding":
                result = run_code_in_sandbox(user_answer, q["test_cases"])
                passed = result.get("passed", 0)
                total = result.get("total", 1)
                score = passed / total
                st.write("Test Results:", result)
            else:
                st.error("Invalid question type for evaluation.")

            st.session_state.score = score
            st.session_state.step = "update_beliefs"
            st.rerun()
        except Exception as e:
            st.error(f"Error evaluating answer: {e}")

# === Step 4: Update Beliefs ===
elif st.session_state.step == "update_beliefs":
    tag = st.session_state.current_tag
    score = st.session_state.score
    try:
        updated_beliefs = update_beliefs(tags=[tag], score=score)
        st.session_state.beliefs = updated_beliefs
        st.success(f"Your score for this question: {round(score * 100, 2)}%")
        if st.button("Ask another question"):
            st.session_state.step = "choose_tag"
            st.rerun()
        if st.button("Finish Evaluation"):
            st.session_state.step = "summarize"
            st.rerun()
    except Exception as e:
        st.error(f"Failed to update beliefs: {e}")

# === Step 5: Summarize Results ===
elif st.session_state.step == "summarize":
    try:
        summary = summarize_results(st.session_state.beliefs)
        st.subheader("📊 Final Summary")
        st.markdown(summary)
        st.write("Belief Scores:", st.session_state.beliefs)
        if st.button("Restart"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
    except Exception as e:
        st.error(f"Failed to summarize results: {e}")

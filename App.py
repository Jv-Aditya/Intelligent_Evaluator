import streamlit as st
import time
import json
from Actions import *
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
from collections import Counter
from sentence_transformers import SentenceTransformer, util
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components
# === LLM Client Setup ===
load_dotenv()
hf_token = os.getenv("hf_token")
client = InferenceClient(provider="fireworks-ai", api_key=hf_token)

# === UI Setup ===
st.set_page_config(page_title="🧠 Intelligent Evaluator", layout="centered")
st.title("🧠 Intelligent Evaluator (LLM-assisted Flow)")

# === Session Initialization ===
if "topic" not in st.session_state:
    st.session_state.topic = ""
if "tags" not in st.session_state:
    st.session_state.tags = []
if "beliefs" not in st.session_state:
    st.session_state.beliefs = {}
if "question" not in st.session_state:
    st.session_state.question = {}
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "max_questions" not in st.session_state:
    st.session_state.max_questions = 10
if "step" not in st.session_state:
    st.session_state.step = "start"
if "question_counts" not in st.session_state:
    st.session_state.question_counts = {}

# === LLM Helper ===
def call_llm_for_next_question(tags, beliefs, asked_types):
    type_counts = Counter(asked_types)
    total_asked = len(asked_types)
    max_questions = st.session_state.max_questions
    mcq_count = type_counts.get("MCQ", 0)
    short_answer_count = type_counts.get("ShortAnswer", 0)
    coding_count = type_counts.get("Coding", 0)
    print(mcq_count,short_answer_count,coding_count)
    system_prompt = f"""
You are an intelligent evaluator tasked with generating a high-quality question to assess a student's understanding of a technical topic (e.g., Python).

Use the following constraints:
- Use only the provided list of tags.
- Choose one or more tags that have not yet been assessed.
- The question should ideally combine multiple related tags in one prompt to evaluate multiple areas at once.
- Choose only from these types: "MCQ", "ShortAnswer", or "Coding".
- Use the "difficulty" field to adapt based on belief strength: start easier for unknown topics, or increase difficulty if belief is high.
- Create only one question.

Important Distribution Rule:
- This test will consist of a total of {max_questions} questions.
- Questions should be distributed approximately as:
    • 50% MCQ → ~{round(0.5 * max_questions)} questions
    • 30% ShortAnswer → ~{round(0.3 * max_questions)} questions
    • 20% Coding → ~{round(0.2 * max_questions)} questions
- Total questions asked so far: {total_asked}
- Already asked: {mcq_count} MCQ, {short_answer_count} ShortAnswer, {coding_count} Coding
- Try to maintain this distribution when generating the next question.

You must return your next question in strict JSON format using the following structure:
{{
  "tags": ["list", "of", "tags"],
  "type": "MCQ" | "ShortAnswer" | "Coding",
  "difficulty": "easy" | "medium" | "hard"
}}

Only return the JSON object. Do not include any commentary, explanation, or markdown formatting.
""".strip()

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": json.dumps({
            "tags": tags,
            "beliefs": beliefs,
            "asked_types": asked_types
        })}
    ]
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=messages
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Failed to parse LLM response: {e}")
        return {}

# === Step 1: Enter Topic ===
if st.session_state.step == "start":
    topic = st.text_input("Enter topic to evaluate:", value="Python")
    if st.button("Start Test"):
        try:
            result = generate_tags(topic)
            st.session_state.topic = topic
            st.session_state.tags = result.get("tags", [])
            st.session_state.beliefs = result.get("beliefs", {})
            st.session_state.asked_types = []
            st.session_state.step = "next_question"
            st.rerun()
        except Exception as e:
            st.error(f"Error generating tags: {e}")

# === Step 2: LLM picks tag/type → generate question ===
elif st.session_state.step == "next_question":
    if st.session_state.question_count >= st.session_state.max_questions:
        st.session_state.step = "summarize"
        st.rerun()
    else:
        decision = call_llm_for_next_question(
            tags=st.session_state.tags,
            beliefs=st.session_state.beliefs,
            asked_types=st.session_state.get("asked_types", [])
        )
        if decision:
            try:
                q = generate_question(
                    tag=decision["tags"],
                    type=decision["type"],
                    difficulty=decision["difficulty"]
                )
                print(q)
                st.session_state.question = q
                st.session_state.current_tag = decision["tags"]
                st.session_state.asked_types.append(decision["type"])
                st.session_state.step = "show_question"
                st.rerun()
            except Exception as e:
                st.error(f"Error generating question: {e}")
        else:
            st.error("LLM failed to suggest a next question.")

# === Step 3: Show Question and Capture Answer ===
# === Step 3: Show Question and Capture Answer ===
elif st.session_state.step == "show_question":
    q = st.session_state.question
    st.subheader(f"Question {st.session_state.question_count + 1}")
    st.markdown(f"**{q['question']}**")

    # === Timer Setup ===
    if "question_start_time" not in st.session_state:
        st.session_state.question_start_time = time.time()

    question_duration = q["time_limit"]
    elapsed = int(time.time() - st.session_state.question_start_time)
    remaining = max(question_duration - elapsed, 0)

    # === JavaScript Countdown Timer (displays but logic is backend-controlled) ===
    import streamlit.components.v1 as components
    components.html(f"""
        <div id="timer" style="font-size:20px; color:#336699; margin-bottom: 10px;"></div>
        <script>
          let countdown = {remaining};
          let timerElement = document.getElementById("timer");

          function updateTimer() {{
            let minutes = Math.floor(countdown / 60);
            let seconds = countdown % 60;
            timerElement.innerHTML = "⏳ Time Remaining: " + 
              String(minutes).padStart(2, '0') + ":" + 
              String(seconds).padStart(2, '0');
            countdown--;
            if (countdown < 0) {{
              timerElement.innerHTML = "⏰ Time is up!";
              clearInterval(timer);
            }}
          }}
          updateTimer();
          let timer = setInterval(updateTimer, 1000);
        </script>
    """, height=50)

    # === Logic Control for Disabling Inputs ===
    time_up = remaining <= 0

    # === Input Setup ===
    user_answer = None
    if st.session_state.get("flag", False):
        if q["type"] == "MCQ":
            st.session_state["mcq_answer"] = ""
        elif q["type"] == "ShortAnswer":
            st.session_state["short_answer"] = ""
        elif q["type"] == "Coding":
            st.session_state["coding_answer"] = ""
        st.session_state.flag = False

    if q["type"] == "MCQ":
        if "mcq_answer" in st.session_state and st.session_state.mcq_answer not in q["options"]:
            del st.session_state["mcq_answer"]
        user_answer = st.radio("Choose your answer:", q["options"], key="mcq_answer", disabled=time_up)

    elif q["type"] == "ShortAnswer":
        if "short_answer" in st.session_state:
            del st.session_state["short_answer"]
        user_answer = st.text_input("Enter your answer:", key="short_answer", disabled=time_up)

    elif q["type"] == "Coding":
        if "coding_answer" in st.session_state:
            del st.session_state["coding_answer"]
        user_answer = st.text_area("Write your code:", height=200, key="coding_answer", disabled=time_up)

    st.session_state.flag = True

    # === Submission Buttons ===
    col1, col2 = st.columns([1, 1])
    submitted = col1.button("✅ Submit Answer", disabled=time_up)
    skipped = col2.button("⏭️ Skip Question")

    if time_up:
        st.warning("⏰ Time is up! You can only skip this question.")

    if skipped:
        st.session_state.beliefs = update_beliefs(tags=st.session_state.current_tag, score=0.0)
        st.success("Question skipped. Moving to the next one.")
        st.session_state.question_count += 1
        st.session_state.flag = True
        st.session_state.step = "next_question"
        st.session_state.pop("question_start_time", None)
        st.rerun()

    if submitted and not time_up:
        try:
            score = 0
            if q["type"] == "MCQ":
                score = evaluate_mcq([user_answer], q["correct_answer"])
            elif q["type"] == "ShortAnswer":
                score = float(evaluate_short_answer(user_answer, q["correct_answer"]))
            elif q["type"] == "Coding":
                result = run_code_in_sandbox(user_answer, q["test_cases"])
                passed = result.get("passed", 0)
                total = result.get("total", 1)
                score = passed / total
                st.write("Code Result:", result)

            for tag in st.session_state.current_tag:
                st.session_state.question_counts[tag] += 1
            st.session_state.question_count += 1
            st.session_state.step = "next_question"
            updated = update_beliefs(tags=st.session_state.current_tag, score=score)
            st.session_state.beliefs = updated
            st.success("✅ Submitted successfully")
            st.session_state.flag = True
            st.session_state.pop("question_start_time", None)
            st.rerun()
        except Exception as e:
            st.error(f"Error during evaluation: {e}")



# === Step 4: Summary ===
elif st.session_state.step == "summarize":
    try:
        summary = summarize_results(st.session_state.beliefs)
        st.subheader("🧠 Final Evaluation Summary")
        st.markdown(summary)
        st.write("Beliefs:", st.session_state.beliefs)

        if st.button("Restart"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
    except Exception as e:
        st.error(f"Failed to summarize results: {e}")

import json
from huggingface_hub import InferenceClient
import streamlit as st
import os

# === Setup Inference Client ===
client = InferenceClient(provider="fireworks-ai", api_key=os.getenv("hf_token"))


def generate_tags(topic: str):
    if "beliefs" not in st.session_state: # dividing the topic into subtopics first time
        prompt = f"""
    You are a helpful assistant designed to break down a learning topic into its core subtopics.
    Given a topic, return a JSON object with two keys: "topic" and "subtopics".

    Requirements:
    - The "topic" key should have the name of the topic.
    - The "subtopics" key should be a list of 5 to 10 relevant subtopics necessary to evaluate knowledge in that topic.
    - Respond ONLY with valid JSON.

    Example:
    Input: Python  
    Output:
    {{
    "topic": "Python",
    "subtopics": ["Data Types", "Control Flow", "Functions", "OOP", "Modules", "File I/O", "Error Handling"]
    }}

    Now generate for topic: {topic}
    """

        try:
            raw_response = query_llm(prompt)
            parsed = json.loads(raw_response)
            subtopics = parsed.get("subtopics", [])

            # Initialize beliefs in session state
            if "beliefs" not in st.session_state:
                st.session_state.beliefs = {}

            if 'question_counts' not in st.session_state:
                st.session_state.question_counts = {}

            for tag in subtopics:
                st.session_state.beliefs[tag] = 0.5 
                st.session_state.question_counts[tag] = 0

            return {
                "topic": topic,
                "tags": subtopics,
                "beliefs": st.session_state.beliefs
            }

        except Exception as e:
            return {
                "error": str(e),
                "message": "Failed to generate tags or parse LLM response."
            }
    else:
        pass
        # return some tags 

def generate_question(tag: str, type: str, difficulty: str):
    return {
        "question": f"What is a {tag} in Python?",
        "options": ["A", "B", "C", "D"] if type == "MCQ" else [],
        "type": type,
        "difficulty": difficulty,
        "time_limit": 60
    }

def evaluate_mcq(choosen_answer: list, correct_answer: list):
    # need to count the number of corrrect options choosen
    correct_answer = [x.strip().upper() for x in correct_answer]
    score = 0
    for i in choosen_answer:
        if i.strip().upper() in correct_answer:
            score += 1
    return score/len(correct_answer)

def evaluate_short_answer(question: str, answer: str):
    return "loop" in answer.lower()

def run_code_in_sandbox(code: str, testcases: list):
    return {"passed": 3, "failed": 0}

def update_beliefs(tags: list, score: float):
    for tag in tags:
        n = st.session_state.question_counts[tag]
        current_belief = st.session_state.beliefs[tag]

        # Running mean formula
        new_belief = (current_belief * n + score) / (n + 1)

        # Store updated values
        st.session_state.beliefs[tag] = new_belief
        st.session_state.question_counts[tag] = n + 1

    return st.session_state.beliefs

def summarize_results():
    return "User has strong knowledge in Loops and Functions."

def query_llm(prompt: str) -> str:
    try:
        response = client.text_generation(
            model="accounts/fireworks/models/llama-v3-8b-instruct",
            prompt=prompt.strip(),
            max_new_tokens=200,
            temperature=0.3
        )
        return response.strip()
    except Exception as e:
        raise RuntimeError(f"LLM query failed: {e}")



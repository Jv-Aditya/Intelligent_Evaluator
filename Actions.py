import json
from huggingface_hub import InferenceClient
import streamlit as st
import os
from dotenv import load_dotenv
import re
from sentence_transformers import SentenceTransformer, util


load_dotenv()
# === Setup Inference Client ===
client = InferenceClient(provider="fireworks-ai", api_key=os.getenv("hf_token"))
def query_llm(prompt):
    # prompt = json.dumps({"messages": messages, "actions": actions or []})
    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": prompt}
        ],
    )
    return completion.choices[0].message["content"]


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

def generate_question(tag: list,type: str, difficulty: str = "medium"):
    prompt = f"""
You are a helpful assistant designed to generate **one** Python assessment question based on the given topics and type and difficulty.
Inputs :
- topics: {tag}          
- type: {type}           
- difficulty: "{difficulty}"

Specifications:
- Question difficulty should match the given difficulty.
- Time limits:
    • MCQ or ShortAnswer → time_limit = 120
    • Coding → time_limit = 600

Output:
Respond **only** with a single JSON object which contain only one questiions using *exactly* this structure and not markdown:
If type == "MCQ":
{{
  "question": "<string>",
  "options": ["<opt1>", "<opt2>", "<opt3>", "<opt4>"],
  "type": "MCQ",
  "correct_answer": "<index as string, e.g. \"0\" or \"2\">",
  "time_limit": 120
}}

If type == "ShortAnswer:
    {{
    "question": "<string>",
    "options": [],
    "type": "ShortAnswer",
    "correct_answer":"<model answer: 1–2 sentences>",
    "time_limit": '120'
    }}
If type == "Coding":
    {{
    "question": "<string>",
    "options": [],
    "type": "Coding",
    "correct_answer": [
        {{
            "input": <literal or list/tuple>,
            "expected_output": <literal or list/tuple>
        }},
        // 'include at least 10–20 test cases'
        ]
    ,
    "time_limit": '600'
    }}

"""
    
    # prompt = generate_questions_prompt(tag, difficulty)
    
    try:
        raw_response = query_llm(prompt)
        cleaned_response = re.sub(r"^```json\s*|\s*```$", "", raw_response.strip())
        questions = json.loads(cleaned_response)
        return questions
    except Exception as e:
        raise ValueError(f"Failed to parse LLM response: {e}\nRaw:\n{raw_response}")



def evaluate_mcq(choosen_answer: list, correct_answer: list):
    # need to count the number of corrrect options choosen
    correct_answer = [x.strip().upper() for x in correct_answer]
    score = 0
    for i in choosen_answer:
        if i.strip().upper() in correct_answer:
            score += 1
    return score/len(correct_answer)

def evaluate_short_answer(user_answer: str, correct_answer: str) -> int:
    """
    Compares two text answers and returns:
    - 1 if semantic similarity >= 0.5
    - 0 if similarity < 0.5
    """
    # Compute embeddings
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = _model.encode([user_answer, correct_answer], convert_to_tensor=True)
    # Calculate cosine similarity
    sim_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    
    # Debug: print score if needed
    # print(f"Similarity score: {sim_score:.3f}")
    return f"{sim_score:.3f}"
    # return 1 if sim_score >= 0.5 else 0

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

def summarize_results(beliefs: dict):
    strong_knowledge = [tag for tag, belief in beliefs.items() if belief > 0.7]
    moderate_knowledge = [tag for tag, belief in beliefs.items() if 0.3 < belief <= 0.7]
    weak_knowledge = [tag for tag, belief in beliefs.items() if belief <= 0.3]

    return f"User has strong knowledge in {', '.join(strong_knowledge)} and User has moderate knowledge in {', '.join(moderate_knowledge)} and User has weak knowledge in {', '.join(weak_knowledge)}."

# def query_llm(prompt: str) -> str:
#     try:
#         response = client.text_generation(
#             model="accounts/fireworks/models/llama-v3-8b-instruct",
#             prompt=prompt.strip(),
#             max_new_tokens=200,
#             temperature=0.3
#         )
#         return response.strip()
#     except Exception as e:
#         raise RuntimeError(f"LLM query failed: {e}")


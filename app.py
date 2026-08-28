import re
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from PIL import Image

app = Flask(__name__)

# ==============================
# GEMINI API (DIRECT CONFIGURATION)
# ==============================
# Paste your Gemini API key directly here
GEMINI_API_KEY = "AQ.Ab8RN6IOp65fJDOPb0tZQ3SGpHEd-aJFOo6Qtlwcu6P-VKUR0A"

genai.configure(api_key=GEMINI_API_KEY.strip())
gemini_model = genai.GenerativeModel("gemini-3.6-flash")

# Semantic model
model = SentenceTransformer('all-MiniLM-L6-v2')


# ==============================
# OCR USING GEMINI
# ==============================
def extract_text_from_image(image_file):
    img = Image.open(image_file)
    prompt = "Extract all text exactly. Keep format like: 1) answer"
    response = gemini_model.generate_content([prompt, img])
    return response.text


# ==============================
# NORMALIZE
# ==============================
def normalize(text):
    return re.sub(r'\s+', ' ', text.lower().strip())


# ==============================
# EXTRACT NUMBER
# ==============================
def extract_number(text):
    nums = re.findall(r'-?\d+\.?\d*', text)
    return float(nums[0]) if nums else None


# ==============================
# SEMANTIC SIMILARITY
# ==============================
def semantic_similarity(a, b):
    emb1 = model.encode(a, convert_to_tensor=True)
    emb2 = model.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2))


# ==============================
# EVALUATE ANSWER
# ==============================
def evaluate_answer(correct, student, max_marks):
    correct_n = normalize(correct)
    student_n = normalize(student)

    if not student_n:
        return 0, "No answer"

    # Maths handling
    c_num = extract_number(correct_n)
    s_num = extract_number(student_n)

    if c_num is not None and s_num is not None:
        if abs(c_num - s_num) < 1e-6:
            return max_marks, "Correct numerical answer"
        else:
            return 0, "Wrong numerical answer"

    # Exact match
    if correct_n == student_n:
        return max_marks, "Exact match"

    # Semantic similarity
    sim = semantic_similarity(correct_n, student_n)

    if max_marks == 1:
        return (1, "Meaning matches") if sim >= 0.85 else (0, "Incorrect")
    else:
        if sim >= 0.85:
            return 2, "Correct meaning"
        elif sim >= 0.65:
            return 1.5, "Mostly correct"
        elif sim >= 0.4:
            return 1, "Partially correct"
        elif sim >= 0.2:
            return 0.5, "Slightly correct"
        else:
            return 0, "Incorrect"


# ==============================
# PARSE ANSWERS
# ==============================
def parse_answers(text):
    pattern = r'(\d+)[\)\.-]\s*([^\n]+)'
    matches = re.findall(pattern, text)
    return {int(q): ans.strip() for q, ans in matches}


# ==============================
# MARK RANGE
# ==============================
def parse_marks_range(one_range, two_range, total_q):
    marks = [0] * total_q

    def fill(rng, val):
        try:
            start, end = map(int, rng.split('-'))
            for i in range(start, end + 1):
                if 1 <= i <= total_q:
                    marks[i - 1] = val
        except Exception:
            pass

    fill(one_range, 1)
    fill(two_range, 2)

    return marks


# ==============================
# ROUTE
# ==============================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Upload images
        answer_key_img = request.files['answer_key']
        student_img = request.files['student']

        # OCR
        answer_key_text = extract_text_from_image(answer_key_img)
        student_text = extract_text_from_image(student_img)

        # Parse
        answer_key = parse_answers(answer_key_text)
        student_answers = parse_answers(student_text)

        print("\n===== ANSWER KEY =====")
        for k, v in answer_key.items():
            print(f"{k}) {v}")

        print("\n===== STUDENT ANSWERS =====")
        for k, v in student_answers.items():
            print(f"{k}) {v}")

        total_q = len(answer_key)

        # Marks
        one_range = request.form['one_mark']
        two_range = request.form['two_mark']
        marks_list = parse_marks_range(one_range, two_range, total_q)

        # Evaluate
        results = []
        total = 0
        max_total = sum(marks_list)

        for i in sorted(answer_key.keys()):
            current_max = marks_list[i - 1] if (i - 1) < len(marks_list) else 1
            score, reason = evaluate_answer(
                answer_key[i],
                student_answers.get(i, ""),
                current_max
            )

            total += score
            results.append({
                "q": i,
                "score": score,
                "max": current_max,
                "reason": reason
            })

        return render_template(
            "result.html",
            results=results,
            total=total,
            max_total=max_total
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)

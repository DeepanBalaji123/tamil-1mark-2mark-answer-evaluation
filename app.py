import os
import re
from flask import Flask, render_template, request
import google.generativeai as genai
from PIL import Image

app = Flask(__name__)

# ==============================
# GEMINI API CONFIGURATION
# ==============================
API_KEY = "AQ.Ab8RN6KnwqlpibPBO3O950exM96CdzdMORXWKUsJiN8WDPs4tg"

genai.configure(api_key=API_KEY.strip())
gemini_model = genai.GenerativeModel("gemini-3.6-flash")


# ==============================
# OCR EXTRACTION
# ==============================
def extract_text_from_image(image_file):
    img = Image.open(image_file)
    prompt = "Extract all text exactly. Keep numbered format strictly like: 1) answer text"
    response = gemini_model.generate_content([prompt, img])
    return response.text


# ==============================
# LIGHTWEIGHT SEMANTIC EVALUATION VIA GEMINI
# ==============================
def evaluate_answer(correct, student, max_marks):
    if not student.strip():
        return 0, "No answer"

    # Exact match
    if correct.strip().lower() == student.strip().lower():
        return max_marks, "Exact match"

    # Prompt Gemini for semantic scoring
    eval_prompt = f"""
    Evaluate the following student's answer against the correct answer key.
    
    Correct Answer: {correct}
    Student Answer: {student}
    Maximum Marks: {max_marks}
    
    Rules:
    - If the meaning matches accurately (even in Tamil or slightly different phrasing), award full marks.
    - If partially correct, award proportionate marks (e.g., 0.5, 1, 1.5).
    - If incorrect, award 0.
    
    Output strictly in this format:
    Score: <numerical score>
    Reason: <brief 2-4 word explanation>
    """
    
    try:
        res = gemini_model.generate_content(eval_prompt).text
        score_match = re.search(r"Score:\s*([0-9.]+)", res)
        reason_match = re.search(r"Reason:\s*(.+)", res)
        
        score = float(score_match.group(1)) if score_match else 0
        reason = reason_match.group(1).strip() if reason_match else "Evaluated"
        return min(score, max_marks), reason
    except Exception:
        return 0, "Evaluation error"


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
        if 'answer_key' not in request.files or 'student' not in request.files:
            return "Please upload both images.", 400

        answer_key_img = request.files['answer_key']
        student_img = request.files['student']

        answer_key_text = extract_text_from_image(answer_key_img)
        student_text = extract_text_from_image(student_img)

        answer_key = parse_answers(answer_key_text)
        student_answers = parse_answers(student_text)

        total_q = len(answer_key)
        one_range = request.form.get('one_mark', '')
        two_range = request.form.get('two_mark', '')
        marks_list = parse_marks_range(one_range, two_range, total_q)

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

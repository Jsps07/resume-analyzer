from flask import Flask, render_template, request
import fitz
import pickle
import pandas as pd
import re

KEYWORDS = ['python', 'machine learning', 'data science', 'deep learning', 'flask',
            'django', 'git', 'sql', 'nlp', 'tensorflow']

app = Flask(__name__)

with open("resume_model.pkl", 'rb') as f:
    model = pickle.load(f)


def extract_certificates(text):
    pattern = r'\b(certified|certification|certificate)\b'
    if re.search(pattern, text.lower()):
        return 1
    return 0

def extract_experience(text):
    matches = re.findall(r'(\d+)\s*(\+)?\s+years?', text.lower()) # returns tuple of all the experience
    if matches:
        years_list = [int(match[0]) for match in matches]
        return max(years_list)
    return 0

def extract_projects(text):
    matches = re.findall(r'(\d+)\s*(\+)?\s*projects?', text.lower())
    if matches:
        project_list = [int(match[0]) for match in matches]
        return max(project_list)
    return 0

def compute_strength(cert, experience_years, projects_count):
    
    certificate = cert

    # Experience assigning
    if experience_years <=2:
        exp_points = 0
    elif 2 < experience_years <=5:
        exp_points = 1
    else:
        exp_points = 2

    # Project points
    if projects_count <=3:
        proj_points = 0
    elif 4 <= projects_count <=6:
        proj_points = 1
    else:
        proj_points = 2

    return certificate + exp_points + proj_points 

@app.route("/", methods=["GET", "POST"])
def home():
    resume_text = ""
    score = 0
    result = None
    strength = None

    if request.method == "POST":
        uploaded_file = request.files['resume']

        # Extract all text from the file if it is a pdf
        if uploaded_file.filename.endswith('.pdf'):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page in doc:
                resume_text += page.get_text()
    

        cert_found = extract_certificates(resume_text)
        years_exp = extract_experience(resume_text)
        proj_count = extract_projects(resume_text)


        score = compute_strength(cert_found, years_exp, proj_count)

        # Calling model
        sample = pd.DataFrame({
            'Skills': [resume_text],
            'Strength': [score]
        })
        
        sample.columns = sample.columns.astype(str)

        prediction = model.predict(sample)[0]

        result = "Hire" if prediction == 1 else "Reject"



    return render_template('index.html', text = resume_text, score = score, result = result)

if __name__ == '__main__':
    app.run(debug=True)
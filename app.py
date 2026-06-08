from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

# تحميل الموديل
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# تنظيف النص
def clean_text(text):
    text = str(text)
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub(r'(.)\1+', r'\1', text)
    text = re.sub(r'[^\u0621-\u064A\s]', '', text)
    return text.strip()

# أنواع المشاكل
ISSUE_KEYWORDS = {
    "performance": ["بطيء", "بطء", "يعلق", "هنق", "يفصل"],
    "usability": ["صعب", "معقد", "غير واضح"],
    "bugs": ["مشكلة", "خطأ", "تعطل", "كراش"],
    "pricing": ["غالي", "سعر", "مكلف"],
    "access": ["ما يفتح", "ما يدخل", "ما يشتغل"]
}

# التوصيات
ISSUE_SOLUTIONS = {
    "performance": "تحسين سرعة وأداء التطبيق",
    "usability": "تبسيط واجهة المستخدم وتحسين التجربة",
    "bugs": "إصلاح الأخطاء والمشاكل التقنية",
    "pricing": "مراجعة الأسعار وتقديم خيارات أفضل",
    "access": "حل مشاكل الدخول والتشغيل"
}

# الدالة الأساسية
def get_feedback(user_review):
    cleaned = clean_text(user_review)

    review_vector = vectorizer.transform([cleaned])
    prediction = model.predict(review_vector)[0]

    prediction = str(prediction).lower()

    suggestions = []

    if prediction == 'negative':
        label = "Negative"

        for issue, keywords in ISSUE_KEYWORDS.items():
            for word in keywords:
                if word in cleaned:
                    suggestions.append(ISSUE_SOLUTIONS[issue])

        suggestions = list(set(suggestions))

        if not suggestions:
            suggestions = ["تحسين عام في جودة الخدمة"]

    else:
        label = "Positive"

    return label, suggestions

# الصفحة الرئيسية
@app.route('/')
def home():
    return render_template("index.html")

# التوقع
@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    label, suggestions = get_feedback(review)

    return render_template(
        "index.html",
        label=label,
        suggestions=suggestions
    )

if __name__ == "__main__":
    app.run(debug=True)
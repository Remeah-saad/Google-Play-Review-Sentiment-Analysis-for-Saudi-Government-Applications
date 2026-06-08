import joblib
import re

# تحميل
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# نفس التنظيف
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

# نفس الاقتراحات
FIX_SUGGESTIONS = {
    "بطيء": "Improve speed and performance",
    "صعب": "Make interface easier",
    "مشكلة": "Fix technical issues",
    "تعطل": "Fix crashes",
    "خطأ": "Fix bugs",
    "غالي": "Review pricing"
}

def predict_and_recommend(review):
    cleaned = clean_text(review)

    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]

    suggestions = []

    if pred == 'negative':
        for word, fix in FIX_SUGGESTIONS.items():
            if word in cleaned:
                suggestions.append(fix)

        if not suggestions:
            suggestions.append("General review needed")

    return pred, suggestions
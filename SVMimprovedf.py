import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import warnings

warnings.filterwarnings('ignore')
# Step 1: Load Dataset
data = pd.read_csv("saudi_reviews_dataset.csv")
data = data.dropna(subset=['content', 'sentiment'])
# Arabic Word Lists
NEGATIVE_WORDS = [
    'بطيء', 'تعليق', 'يعلق', 'ما يفتح', 'ما يشتغل',
    'تعطل', 'كراش', 'مشكلة', 'خطأ', 'سيء',
    'صعب', 'معقد', 'يفصل', 'هنق', 'ما يدخل'
]
POSITIVE_WORDS = [
    'ممتاز', 'رائع', 'جيد', 'سهل', 'مفيد', 'شكراً',
    'أحسن', 'يستاهل', 'سريع', 'جميل', 'نظيف',
    'احترافي', 'مريح'
]
# Text Cleaning
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
# Feature Engineering
def engineer_features(df):
    df['review_length'] = df['cleaned_content'].apply(len)
    df['word_count'] = df['cleaned_content'].apply(lambda x: len(x.split()))
    df['neg_word_count'] = df['content'].apply(
        lambda x: sum(1 for w in NEGATIVE_WORDS if w in str(x))
    )
    df['pos_word_count'] = df['content'].apply(
        lambda x: sum(1 for w in POSITIVE_WORDS if w in str(x))
    )
    df['sentiment_signal'] = df['pos_word_count'] - df['neg_word_count']

    return df
# Prepare Data
data['cleaned_content'] = data['content'].apply(clean_text)
data = data[data['cleaned_content'] != ""]
data = engineer_features(data)
print("Rows:", len(data))
# Split Data
X = data['cleaned_content']
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    max_df=0.9
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
# Class Weight
classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)
class_weight_dict = dict(zip(classes, weights))
# Grid Search
param_grid = {
    'C': [0.1, 0.5, 1, 10],
    'max_iter': [1000, 2000],
    'loss': ['hinge', 'squared_hinge']
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    LinearSVC(class_weight=class_weight_dict),
    param_grid,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)
print("Running Grid Search...")
grid_search.fit(X_train_vec, y_train)
best_params = grid_search.best_params_
print("Best Parameters:", best_params)
# Final Model
model = LinearSVC(
    C=best_params['C'],
    max_iter=best_params['max_iter'],
    loss=best_params['loss'],
    class_weight=class_weight_dict
)

model.fit(X_train_vec, y_train)

# Evaluation
predictions = model.predict(X_test_vec)
acc = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average='macro')
print("\nAccuracy:", round(acc, 4))
print("Macro F1:", round(f1, 4))
print(classification_report(y_test, predictions))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(
    ['Accuracy', 'Macro F1'],
    [acc, f1],
    width=0.5
)
axes[0].set_ylim(0, 1)
axes[0].set_title("Model Performance")
axes[0].set_ylabel("Score")
for i, v in enumerate([acc, f1]):
    axes[0].text(i, v + 0.02, f"{v:.3f}",
                 ha='center',
                 fontweight='bold')
cm = confusion_matrix(y_test, predictions)
tn, fp, fn, tp = cm.ravel()
labels = np.array([
    [f"TN\n{tn}", f"FP\n{fp}"],
    [f"FN\n{fn}", f"TP\n{tp}"]
])

im = axes[1].imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        axes[1].text(
            j, i,
            labels[i, j],
            ha='center',
            va='center',
            fontsize=14,
            fontweight='bold',
            color='white' if cm[i, j] > cm.max()/2 else 'black'
        )

axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])

axes[1].set_xticklabels(['Negative', 'Positive'])
axes[1].set_yticklabels(['Negative', 'Positive'])

axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
axes[1].set_title("Confusion Matrix")

plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.savefig("model_performance.png", dpi=150)
#plt.show()
# Suggestions System
ISSUE_KEYWORDS = {
    "performance": ["بطيء", "بطء", "يعلق", "هنق", "يفصل"],
    "usability": ["صعب", "معقد", "ما اعرف", "غير واضح"],
    "bugs": ["مشكلة", "خطأ", "تعطل", "كراش"],
    "pricing": ["غالي", "سعر", "مكلف"],
    "access": ["ما يفتح", "ما يدخل", "ما يشتغل"]
}
ISSUE_SOLUTIONS = {
    "performance": "تحسين سرعة وأداء التطبيق",
    "usability": "تبسيط واجهة المستخدم وتحسين التجربة",
    "bugs": "إصلاح الأخطاء والمشاكل التقنية",
    "pricing": "مراجعة الأسعار وتقديم خيارات أفضل",
    "access": "حل مشاكل الدخول والتشغيل"
}
# Corrected Feedback Function
def get_feedback(user_review):
    cleaned = clean_text(user_review)

    review_vector = vectorizer.transform([cleaned])
    prediction = model.predict(review_vector)[0]

    suggestions = []

    if prediction == 'negative':
        for issue, keywords in ISSUE_KEYWORDS.items():
            for word in keywords:
                if word in cleaned:
                    suggestions.append(ISSUE_SOLUTIONS[issue])

        # إزالة التكرار
        suggestions = list(set(suggestions))

        # fallback
        if not suggestions:
            suggestions = ["تحسين عام في جودة الخدمة"]

    return prediction, suggestions

import joblib

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
# data_collection.py
# This file collects Arabic reviews from Google Play Store
# and saves them to a CSV file

import pandas as pd
from google_play_scraper import Sort, reviews

# List of Saudi app package names
app_packages = [

    # --- New Saudi Health & Government Apps ---
    'com.mowaamah',  # موائمة  (ذوي الإعاقة - تكامل)
    'co.ntime.cchi_majlis',  # ضمان يهتم (مجلس الضمان الصحي)
    'sa.gov.nic.tawakkalna',  # توكلنا طوارئ
    'sfda.tamini',  # طمني (هيئة الغذاء والدواء)
    'com.lean.sehhaty',  # صحتي (موجود بالأصل - للتأكيد)
    'sa.gov.nic.myid',  # نفاذ (المركز الوطني للمعلومات)
    'com.mngha.ngha.myvisit',  # بينهم (الحرس الوطني الصحي)
    'moh.gov.sa.mawid',  # موعد (نظام المواعيد الوطني MOH)
    'com.gov.moh.MOHMobile',  # وزارة الصحة (البوابة الموحدة)
    'sa.gov.hrsd.UnifiedApp',  # خدمات الموارد البشرية والتنمية الاجتماعية
]

# Empty list to store all reviews
all_reviews = []

# Loop through each app and collect reviews
for app in app_packages:
    print(f"Collecting reviews for: {app}")

    for score in range(1, 6):  # scores 1 to 5
        try:
            result, _ = reviews(
                app,
                lang='ar',        # Arabic language
                country='sa',     # Saudi Arabia
                sort=Sort.MOST_RELEVANT,
                count=200,        # max 200 reviews per score
                filter_score_with=score
            )

            # Add app name to each review
            for r in result:
                r['appId'] = app

            all_reviews.extend(result)
            print(f"  Score {score}: collected {len(result)} reviews")

        except Exception as e:
            print(f"  Error for {app}, score {score}: {e}")

# Convert to DataFrame
df = pd.DataFrame(all_reviews)

# Remove 3-star reviews (neutral — not clearly positive or negative)
df = df[df['score'] != 3].copy()

# Add sentiment column: 4-5 stars = positive, 1-2 stars = negative
df['sentiment'] = df['score'].apply(
    lambda x: 'positive' if x >= 4 else 'negative'
)

# --- Class Balance Report ---
print("\n--- Class Distribution ---")
print(df['sentiment'].value_counts())
print(f"Balance ratio: {df['sentiment'].value_counts(normalize=True).to_dict()}")

# Save to CSV file
df.to_csv('saudi_reviews_dataset.csv', index=False, encoding='utf-8-sig')

print(f"\nDone! Total reviews collected: {len(df)}")
print("File saved: saudi_reviews_dataset.csv")
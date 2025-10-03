import re, joblib, pandas as pd,os

#Model Loading

model_path = os.path.join(os.getcwd(), "model.joblib")
art = joblib.load(model_path)
model = art['model']
scaler = art['scaler']
feature_cols = art['feature_cols']

spammy_words = ['cash', 'click', 'free', 'money', 'offer', 'prize', 'trial', 'urgent', 'win', 'loto']

def extract_features_from_text_simple(text):
    t = str(text)
    words = re.findall(r"\w+", t)  # Fixed regex
    n_words = max(len(words), 1)
    words_count = n_words
    links_count = len(re.findall(r"https?://|www\.", t, flags=re.IGNORECASE))
    capital_words = sum(1 for w in words if w.isalpha() and w.isupper() and len(w) > 1)
    lower = t.lower()
    spam_word_count = sum(lower.count(sw) for sw in spammy_words)
    return pd.DataFrame([{'words': words_count, 'links': links_count, 'capital_words': capital_words, 'spam_word_count': spam_word_count}], columns=feature_cols)

if __name__ == '__main__':
    text = input("Enter text to check: ")
    feats = extract_features_from_text_simple(text)
    feats_s = scaler.transform(feats)
    pred = int(model.predict(feats_s)[0])
    print('Prediction (1=spam):', pred)

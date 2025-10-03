import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

#CSV
df = pd.read_csv("a_latsuzbaia24_563892.csv")

x = df[['words', 'links', 'capital_words', 'spam_word_count']]
y = df['is_spam']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#Scale
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#Train model
model = LogisticRegression()
model.fit(x_train_scaled, y_train)

#Artifacts
artifacts = {
    "model": model,
    "scaler": scaler,
    "feature_cols": x.columns.tolist()
}
#Generate joblib file
joblib.dump(artifacts, "model.joblib")

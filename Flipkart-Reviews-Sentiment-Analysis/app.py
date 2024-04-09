from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the model and vectorizer
with open('flipkart_sentiment_model.pkl', 'rb') as file:
    loaded_model = joblib.load(file)

with open('tfidf_vectorizer_flipkart_data.pkl', 'rb') as file:
    loaded_vectorizer = joblib.load(file)

@app.route('/')
def home():
    clear_text = request.args.get('clear_text')
    return render_template("home.html", clear_text=clear_text)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    user_review = request.form.get("input_text")

    if not user_review:
        return render_template("prediction.html", user_review="No text provided", sentiment="")

    text_vector = loaded_vectorizer.transform([user_review])
    sentiment = loaded_model.predict(text_vector)

    if sentiment[0] == 0:
        sentiment_text = "Positive Sentiment 😍 🍾 🎉"
    elif sentiment[0] == 1:
        sentiment_text = "Negative Sentiment 🤧 😡 🔥"
    else:
        sentiment_text = "Neutral Sentiment 😶 🙂"

    return render_template("prediction.html", user_review=user_review, sentiment=sentiment_text)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)

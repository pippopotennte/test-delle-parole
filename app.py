import pandas as pd
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
import joblib
import streamlit as st

patterns = {
    r'\d+': '',  # remove digits (numeri)
    r'[^\w\s]': '',  # remove punteggiatura e simboli ...,'@!£$%
    r'\b\w{1,2}\b': '',  # remove all token less than 2 characters
    r'(http|www)[^\s]+': '',  # remove website
    r'\s+': ' '  # sostituisce tutti i multipli spazi con uno spazio
}

def clean_text(text):
    if not isinstance(text, str):
        return ''
    for pattern, replacement in patterns.items():
        text = re.sub(pattern, replacement, text)
    return text.lower()


def main():

    st.write("""
    # NLP
    Detecting Fake News with NLP and Machine Learning
    """)

    data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
    if data is not None:
        df = pd.read_csv(data)
        st.dataframe(df.head())
        
        # Clean the 'text' column
        df['text'] = df['text'].astype(str).apply(clean_text)
        
        if st.button("Train Model"):
            X = df['text']
            y = df['class']

            bow = CountVectorizer(max_features=1500, min_df=5, max_df=0.7)
            tfidf = TfidfTransformer()
            clf = MultinomialNB(alpha=0.1)

            pipe = Pipeline([
                ('bow', bow),
                ('tfidf', tfidf),
                ('clf', clf),
            ])

            pipe.fit(X, y)

            joblib.dump(pipe, 'NLPEs2.pkl')
            st.write('Model trained successfully.')

    model = joblib.load('NLPEs2.pkl')

    st.write("""
    ## Predict Fake or True News
    Enter a news sentence to predict if it's fake or true:
    """)

    user_input = st.text_input("News Sentence")
    if user_input:
        cleaned_input = clean_text(user_input)
        prediction = model.predict([cleaned_input])
        result = 'Fake' if prediction[0] == 0 else 'True'
        st.write(f"The news sentence is predicted as: {result}")

if __name__ == '__main__':
    main()

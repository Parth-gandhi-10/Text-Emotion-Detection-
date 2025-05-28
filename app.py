import streamlit

import sklearn
import pandas as pd
import numpy as np
import altair as alt

import joblib

pipe_lr = joblib.load(open("model/text_emotion.pkl","rb"))
emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happiness": "🤗", "relief": "😂", "fun":"😂", "neutral": "😐", "sad": "😔",
                       "sadness": "😔", "shame": "😳", "surprise": "😮", "love":"❤️","boredom":"🥱","hate":"😡","worry":"😟"  }

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results=pipe_lr.predict_proba([docx])
    return results


def main():
    streamlit.title("Text Emotion Detection")
    streamlit.subheader("Detect Emotions in Text")

    with streamlit.form(key='my_form'):
        raw_text= streamlit.text_area("Typr here")
        submit_text=streamlit.form_submit_button(label='submit')

    if submit_text:
        col1, col2=streamlit.columns(2)

        prediction = predict_emotions(raw_text)
        probability= get_prediction_proba(raw_text)

    with col1:
        streamlit.success("original text")
        streamlit.write(raw_text)
        streamlit.success("prediction")
        emoji_icon = emotions_emoji_dict[prediction]
        streamlit.write("{}:{}".format(prediction, emoji_icon))
        streamlit.write("confidence:{}".format(np.max(probability)))

    with col2:
        streamlit.success("prediction probability")
        #streamlit write proba
        proba_df= pd.DataFrame(probability, columns=pipe_lr.classes_)
        # st.write(proba_df.T)
        proba_df_clean = proba_df.T.reset_index()
        proba_df_clean.columns = ["emotions", "probability"]

        fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
        streamlit.altair_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
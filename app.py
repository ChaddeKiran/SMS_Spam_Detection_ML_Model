#import packages.data_preprocessor as dp
import streamlit as st 
import joblib
import os
os.chdir("C:\\Users\\manis\\OneDrive\\Desktop\\python exercises\\project_NLP\\spam_ml")

spam_clf=joblib.load(open('C:\\Users\\manis\\OneDrive\\Desktop\\python exercises\\project_NLP\\spam_ml\\models\\my_spam_model.pkl','rb'))

vectorizer=joblib.load(open('C:\\Users\\manis\\OneDrive\\Desktop\\python exercises\\project_NLP\\spam_ml\\vectors\\my_vectorizer.pickel','rb'))

def main(title="Spam_SMS_Detection APP".upper()):
    st.markdown(f"<h1 style='text-align: center; font-size: 25px; color: blue;'>{title}</h1>", unsafe_allow_html=True)
    st.image("C:\\Users\\manis\\OneDrive\\Desktop\\python exercises\\project_NLP\\spam_ml\\images\\image.jpg",width=100)
    info = ''

    with st.expander("1. Check if your text spam or not spam "):
        text_message=st.text_input("Please enter your message")
        if st.button("Predict"):
            prediction = spam_clf.predict(vectorizer.transform([text_message]))

            if(prediction[0] == 0):
                info = 'NOT SPAM'

            else:
                info = 'SPAM'
            st.success('Prediction: {}'.format(info))


if __name__ == "__main__":
    main()


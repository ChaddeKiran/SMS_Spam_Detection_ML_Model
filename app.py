#import packages.data_preprocessor as dp
import streamlit as st 
import joblib
#import os
#os.chdir("F:\SMS\SPAM_SMS_DETECTION")

spam_clf=joblib.load(open('/app/models/my_spam_model.pkl','rb'))

vectorizer=joblib.load(open('/app/vectors/my_vectorizer.pickel','rb'))

def main(title="Spam_SMS_Detection APP".upper()):
    st.markdown(f"<h1 style='text-align: center; font-size: 25px; color: blue;'>{title}</h1>", unsafe_allow_html=True)
    st.image("/app/images/image.jpg",width=100)
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


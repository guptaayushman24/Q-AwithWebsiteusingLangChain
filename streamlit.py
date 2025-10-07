import streamlit as st
import requests
import json
def searchForQuestion (user_text:str) :
    try :
        payload = {"query": user_text}
        api_url  = "https://q-awithwebsiteusinglangchain.onrender.com/ask"
        response = requests.post(api_url,json=payload)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e :
        st.error(f"Error calling API: {e}")
        return data
st.title("Question and Answer with the Open AI Wikipedia")

user_text = st.text_area("Enter your question here")
print("User Text is"+" "+user_text)
answer=""
if st.button("Search") :
    if user_text.strip():
        answer = searchForQuestion(user_text)
        if answer:
            st.subheader("Response:")
            st.write(answer)
    else:
        st.warning("Please enter a question before searching.")


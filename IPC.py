import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer, util
import streamlit as st

# Preprocess function
def preprocess_text(text):
    text = text.lower()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    preprocessed_text = ' '.join(words)
    return preprocessed_text

# Load preprocessed data and model
new_ds = pickle.load(open('D:/MyWorkSpace/Summer-Internship-Learn_and_Build/29-July-2024-FIR-Project/preprocess_data.pkl', 'rb'))
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Suggest sections function
def suggest_sections(complaint, dataset, min_suggestions=3):
    preprocessed_complaint = preprocess_text(complaint)
    complaint_embedding = model.encode(preprocessed_complaint)
    section_embedding = model.encode(dataset['Combo'].tolist())
    similarities = util.pytorch_cos_sim(complaint_embedding, section_embedding)[0]
    similarity_threshold = 0.2
    relevant_indices = []
    while len(relevant_indices) < min_suggestions and similarity_threshold > 0:
        relevant_indices = [i for i, sim in enumerate(similarities) if sim > similarity_threshold]
        similarity_threshold -= 0.05
    sorted_indices = sorted(relevant_indices, key=lambda i: similarities[i], reverse=True)
    # Limit the number of suggestions
    suggestions = dataset.iloc[sorted_indices][:min_suggestions][['Description', 'Offense', 'Punishment', 'Cognizable', 'Bailable', 'Court']].to_dict(orient='records')
    return suggestions

# Streamlit app starts here
st.title("IPC Section Suggestion System")

# Input area for complaint description
complaint = st.text_area("Enter Crime Description", height=200)

# Button to get suggestions
if st.button("Get Suggestions"):
    if complaint.strip():
        with st.spinner("Processing..."):
            suggestions = suggest_sections(complaint, new_ds)
            
        if suggestions:
            st.success("Suggested Sections are:")
            for suggestion in suggestions:
                st.markdown(f"""
                **Description**: {suggestion['Description']}  
                **Offense**: {suggestion['Offense']}  
                **Punishment**: {suggestion['Punishment']}  
                **Cognizable**: {suggestion['Cognizable']}  
                **Bailable**: {suggestion['Bailable']}  
                **Court**: {suggestion['Court']}  
                """)
                st.markdown("---")
        else:
            st.error("No matching records found.")
    else:
        st.warning("Please enter a valid crime description.")

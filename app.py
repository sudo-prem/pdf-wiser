import streamlit as st
from tokenizer import tokenizer
from pdfwiser import pdfwiser

# Set the title of the app
st.title('star-tokenizer')

# Create a sidebar with page selection
page = st.sidebar.selectbox("Select Page", ["Tokenizer", "PDF Util"])

# Display the selected page
if page == "Tokenizer":
    tokenizer()
elif page == "PDF Util":
    pdfwiser()

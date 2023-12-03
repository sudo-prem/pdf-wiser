import streamlit as st
from tokenizer import tokenizer
from pdfwiser import pdf_wiser

# Set the default page to "Tokenizer"
page = st.sidebar.selectbox("Select Page", ["Tokenizer", "PDFwiser"], index=0)

# Set the title based on the selected page
if page == "Tokenizer":
    st.title('Tokenizer')
    tokenizer()
elif page == "PDFwiser":
    st.title('PDFwiser')
    pdf_wiser()

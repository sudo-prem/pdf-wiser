import streamlit as st
import base64
from io import BytesIO
import fitz
import tempfile
import os
from embeddings import *


def highlight_and_display_pdf(pdf_bytes, words_to_highlight):
    # Save the PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_filename = temp_file.name
        temp_file.write(pdf_bytes)

    # Open the PDF file using PyMuPDF (fitz)
    pdf_document = fitz.open(temp_filename)

    # Iterate through each page in the PDF
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]

        # Iterate through each word to highlight on the page
        for word in words_to_highlight:
            # Search for the word and get its coordinates
            highlight = page.search_for(word)

            # Add a highlight annotation for each occurrence of the word
            for inst in highlight:
                highlight_annotation = page.add_highlight_annot(inst)
                highlight_annotation.update()

    # Save the modified PDF to a BytesIO object
    modified_pdf_bytes = BytesIO()
    pdf_document.save(modified_pdf_bytes, garbage=4, deflate=True, clean=True)
    pdf_document.close()

    # Display the modified PDF using Streamlit
    display_PDF(modified_pdf_bytes.getvalue())

    # Remove the temporary file
    os.remove(temp_filename)


def display_PDF(pdf_bytes):
    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

    pdf_display = f"""<embed
    class="pdfobject"
    type="application/pdf"
    title="Embedded PDF"
    src="data:application/pdf;base64,{base64_pdf}"
    style="overflow: auto; width: 100%; height: 100em;">"""

    st.markdown(pdf_display, unsafe_allow_html=True)


def pdf_wiser():
    # File uploader widget
    uploaded_file = st.file_uploader(
        "Choose a PDF file", type="pdf", key="pdf_uploader"
    )

    if uploaded_file is not None:
        # Read the file content as bytes
        pdf_bytes = uploaded_file.read()

        words = get_words(uploaded_file, "what did clara stumble upon?")
        # Define a list of words to highlight
        words_to_highlight = [words]
        
        # Highlight the specified words and display the modified PDF
        highlight_and_display_pdf(pdf_bytes, words_to_highlight)

        # Enter user question
        # query = st.chat_input("Enter Question")


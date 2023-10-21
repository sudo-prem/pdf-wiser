import streamlit as st
import tiktoken
from transformers import AutoTokenizer

# Set the title of the app
st.title('star-tokenizer')

# Create a dropdown to select options
option = st.selectbox('Model', ['bert-base-uncased'])

# Create a text input for the user to type something
user_input = st.text_area('Prompt', height=7)  # Set height to 7 lines

# Initialize the token count output
token_count = 0
tokenizer = AutoTokenizer.from_pretrained(option)
tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(user_input)))
st.markdown(f'{tokens}')
token_count = len(tokens)

# Display the token count in the output box
st.markdown(f'**Token Count:** {token_count}')

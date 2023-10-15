import streamlit as st
import tiktoken

# Set the title of the app
st.title('star-tokenizer')

# Create a dropdown to select options
option = st.selectbox('Model', ['Llama 2', 'GPT-3.5'])

# Create a text input for the user to type something
user_input = st.text_input('Prompt')

# Initialize the token count output
token_count = 0

# Calculate token count based on the selected option and user input
if option == 'Llama 2':
    token_count = len(user_input.split())
elif option == 'GPT-3.5':
    token_count = len(user_input.split())

# Display the token count in the output box
st.markdown(f'**Token Count:** {token_count}')

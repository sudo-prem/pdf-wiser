import streamlit as st
from transformers import AutoTokenizer

# Set the title of the app
st.title('star-tokenizer')

# Create a dropdown to select options
option = st.selectbox('Model', ['None', 'bert-base-uncased', 'NousResearch/Llama-2-7b-chat-hf',
                      'mistralai/Mistral-7B-Instruct-v0.1', 'mistralai/Mistral-7B-v0.1'])
if option == 'None':
    model_name = st.text_input("Enter hugging face model name")
else:
    model_name = option

if model_name:
    # Create a text input for the user to type something
    user_input = st.text_area('Prompt', height=512)  # Set height to 7 lines

    # Initialize the token count output
    token_count = 0
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(user_input)))
    st.markdown(f'{tokens}')
    token_count = len(tokens)

    # Display the token count in the output box
    st.markdown(f'**Token Count:** {token_count}')

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

    if user_input:
        # Tokenize the user input
        input_ids = tokenizer.encode(user_input, add_special_tokens=False)

        # Get the token length
        token_count = len(input_ids)

        # Display the token count
        st.markdown(f'Token Count: {token_count}')

        # Display the tokens as well
        st.markdown(f'Tokens: {input_ids}')

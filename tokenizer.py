import streamlit as st
from transformers import AutoTokenizer


def tokenizer():
    # Create a dropdown to select options
    option = st.selectbox('Model', [
        'None',
        'bert-base-uncased',
        'NousResearch/Llama-2-7b-chat-hf',
        'mistralai/Mistral-7B-Instruct-v0.1',
        'mistralai/Mistral-7B-v0.1'
    ])

    if option == 'None':
        model_name = st.text_input("Hugging Face model name")
    else:
        model_name = option

    if model_name:
        # Create a two-column layout
        col1, col2 = st.columns(2)

        with col1:
            # Create a text input for the user to type something
            # Set height to 7 lines
            user_input = st.text_area('Prompt', height=512, key="prompt")

        with col2:
            # Initialize the token count output
            token_count = 0
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            if user_input:
                # Tokenize the user input
                input_ids = tokenizer.encode(
                    user_input, add_special_tokens=False)
                tokens = tokenizer.tokenize(tokenizer.decode(input_ids))

                # Remove one instance of '##' or '_' from the beginning of subword tokens
                for i in range(len(tokens)):
                    if tokens[i].startswith('##'):
                        tokens[i] = tokens[i][2:]
                    elif tokens[i].startswith('▁'):
                        tokens[i] = tokens[i][1:]

                # Get the token length
                token_count = len(input_ids)

                # Display the tokens in the same line with different background colors and justify the content
                colored_tokens = []
                # Customize your background colors
                colors = ['#cce0ff', '#ffccd6', '#daffcc', '#ffd6cc']
                for i, token in enumerate(tokens):
                    # Cycle through background colors
                    color = colors[i % len(colors)]
                    colored_tokens.append(
                        f'<span style="background-color:{color}; color: black; padding: 4px; border-radius: 4px;">{token}</span>')

                # Join colored tokens into a single string with justification
                colored_tokens_str = ' '.join(colored_tokens)
                justified_tokens_str = f'<div style="text-align: justify; color: white; height: 512px; overflow-y: scroll; background-color: white; border-radius: 10px; padding: 10px;">{colored_tokens_str}</div>'

                # Display the token count with a smaller font size
                token_count_text = f'<p style="font-size: 14px; margin-bottom: 4px;">Token Count: {token_count}</p>'
                st.markdown(token_count_text, unsafe_allow_html=True)

                # Display the colored tokens with justification
                st.markdown(justified_tokens_str, unsafe_allow_html=True)

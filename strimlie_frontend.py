import streamlit as st
from langraph_backend import chatbot
from langchain_core.messages import SystemMessage, HumanMessage,BaseMessage

thread_id = '1'
config = {'configurable' : {'thread_id' : thread_id}}
if 'user_history' not in st.session_state:
    st.session_state['user_history'] = []

for message in st.session_state['user_history']:

    with st.chat_message(message['role']):
        st.write(message['content'])

user_message = st.chat_input('Type here')

if user_message:
    st.session_state['user_history'].append({'role' : 'user', 'content': user_message})
    with st.chat_message('user'):
        st.write(user_message)
    
    
    # response  = chatbot.invoke({'messages' : [HumanMessage(content =  user_message)]},config = config)
    # ai_message = response['messages'][-1].content[0]['text']
    

    with st.chat_message('assistant'):
        ai_message = st.write_stream(
                message_chunk.content for message_chunk, metadata in chatbot.stream(
                    {'messages' : [HumanMessage(content =  user_message)]},
                    config = config,
                    stream_mode = 'messages'
            )
        )   

    st.session_state['user_history'].append({'role' : 'assistant', 'content': ai_message})    
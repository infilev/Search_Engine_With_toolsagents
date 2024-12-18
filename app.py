import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

## Used for exceptions for handling retrying requests
from duckduckgo_search.exceptions import RatelimitException

import os
from dotenv import load_dotenv

load_dotenv()

## TOOLS

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max= 200)
wiki = WikipediaQueryRun(api_wrapper = api_wrapper_wiki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max= 200)
arxiv = ArxivQueryRun(api_wrapper = api_wrapper_arxiv)

search =DuckDuckGoSearchRun(name = "Search")

st.title("Langchain- Chat with Simple Search using Tool and Agent")

"""
In this example we are using StreamlitCallbackHandler to display the thoughts and actions
"""

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")


# Chat history 

if "messages" not in st.session_state:
        st.session_state["messages"] = [

            {
                "role": "user", "content": "Hi! I'm a chatbot who can search the web. How can I help you?."
            },
        ]

for msg in st.session_state.messages:
    st.chat_message (
        msg["role"]).write(msg['content']
        
    )
          

if prompt:=st.chat_input(placeholder= "What is machine learning"):
    st.session_state.messages.append(
        {
            "role": "user", "content": prompt
        }  

    )
    st.chat_message("user").write(prompt)
    llm = ChatGroq(groq_api_key=api_key, model = "Llama3-8b-8192", streaming = True)
    tools = [search, arxiv, wiki]
    
    search_agent = initialize_agent(tools, llm, agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_error = True)




    # Retry logic for handling rate limits
    def get_response_with_retry():
        retries = 3  # Number of retries
        for attempt in range(retries):
            try:
                # Run the agent and get response
                return search_agent.run(st.session_state.messages, callbacks=[StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)])
            except RatelimitException as e:
                # Handle rate limit exceptions
                st.write(f"Rate limit reached. Retrying... ({attempt + 1}/{retries})")
                time.sleep(5)  # Wait before retrying
            except Exception as e:
                # General error handling
                st.write(f"Error occurred: {str(e)}")
                return None
        return "Unable to get a response after retries."



## To display the chain of thoughts
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(
            st.container(),
            expand_new_thoughts= False
            )

        response = search_agent.run(
            st.session_state.messages,
            callbacks=[st_cb]
        )    

        st.session_state.messages.append(
            {
                "role": "assistant", "content": response
            }
        )
        st.write(response)

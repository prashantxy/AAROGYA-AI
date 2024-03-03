import streamlit as st
import huggingface_hub
from langchain_community.llms import huggingface_hub
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers


## to get response
def getLLamaresponse(disease,input_text,no_words):
    
    ### LLama2 model\

    llm=CTransformers(model='AARAGYA AI\llama-2-7b-chat.ggmlv3.q4_K_S.bin',
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature':0.01})
    
    ## Prompt Template
    
    template="""
    ASK YOUR QUERRIES FOR {disease} disease for a topic {input_text}
    within {no_words} words.
    """
    prompt=PromptTemplate(input_variable=["disease","input_text",'no_words'],
                          template=template)
    
    ## Generate the response
    response=llm(prompt. format(disease=disease,input_text=input_text,no_words=no_words))
    print(response)
    return response


st.set_page_config(page_title='AI AAROGYA',
                   
                   layout='centered',
                   initial_sidebar_state = 'collapsed')

st.header("AI AAROGYA")

input_text=st.text_input("Ask your Queries")

## creating columns

col1,col2=st.columns([5,5])

with col1:
    no_words=st.text_input('No of words')
    with col2:
        disease=st.selectbox('ASK THE QUERIES FOR ANY MEDICAL ISSUE',
                                ('INFANT','CHILD','ADULT','MIDDLEAGE','OLD AGE',),index=0)
        submit=st.button("Generate")
        
        ##final response
        if submit:
            st.write(getLLamaresponse(disease,input_text,no_words))
        

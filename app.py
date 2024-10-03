from langchain import HuggingFaceHub
from langchain import PromptTemplate
from langchain import LLMChain
import os
import chainlit as cl

model_id = "gpt2-medium"
# Instantiate the HuggingFaceHub class with the correct parameters
HUGGINGFACEHUB_API_TOKEN = "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"
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN
conv_model = HuggingFaceHub(huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN'], repo_id=model_id)
prompt_template = """You are a helpful AI assistant that makes the proper stories by completing the query provided by the user
{query}
"""
@cl.on_chat_start ##entry point of the chatbot
def main():
    max_length = 300
    prompt = PromptTemplate(template=prompt_template, input_variables=['query'])
    conv_model = HuggingFaceHub(huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN'], 
                            repo_id=model_id,
                            model_kwargs={"max_length": max_length})
    conv_chain = LLMChain(llm=conv_model, prompt=prompt, verbose=True) #conversational chain
    cl.user_session.set("llm_chain",conv_chain)

    # print(conv_chain.run("Once upon a time in 1947, a mysterious event unfolded that changed the course of history. .."
    # ))
@cl.on_message
async def main(message:str):
    llm_chain = cl.user_session.get("llm_chain")
    res = await llm_chain.acall(message,callbacks= [cl.AsyncLangchainCallbackHandler()])   #acall is anysynchronous
    await cl.Message(content=res["text"].send())


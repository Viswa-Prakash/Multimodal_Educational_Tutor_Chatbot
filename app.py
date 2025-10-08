import streamlit as st
from PIL import Image
import easyocr
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import os
from dotenv import load_dotenv
import speech_recognition as sr
from pydub import AudioSegment

# --- Configuration and Model Loading ---
load_dotenv()

if "OPENAI_API_KEY" not in os.environ:
    st.error("OpenAI API key not found. Please add it to your .env file.")
    st.stop()

@st.cache_resource
def load_models():
    """Loads all the required models and returns them as a dictionary."""
    print("Loading models...")
    ocr_reader = easyocr.Reader(['en'])
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7)
    print("Models loaded successfully.")
    return {
        "ocr": ocr_reader,
        "blip_processor": blip_processor,
        "blip_model": blip_model,
        "llm": llm
    }

models = load_models()
llm = models["llm"]

# --- LangGraph State and Nodes ---
class TutorState(TypedDict):
    messages: List[BaseMessage]
    topic: str
    sub_topics: List[str]
    current_step: int
    context_summary: str
    user_question: str

def process_image(image_file):
    img = Image.open(image_file).convert('RGB')
    ocr_result = models["ocr"].readtext(image_file.getvalue(), detail=0, paragraph=True)
    ocr_text = " ".join(ocr_result)
    inputs = models["blip_processor"](img, return_tensors="pt")
    out = models["blip_model"].generate(**inputs, max_new_tokens=50)
    img_description = models["blip_processor"].decode(out[0], skip_special_tokens=True)
    return f"Image Description: {img_description}. Extracted Text: {ocr_text}"

def process_audio(audio_file):
    
    r = sr.Recognizer()
    try:
        audio_segment = AudioSegment.from_file(audio_file)
        audio_segment.export("temp_audio.wav", format="wav")
        with sr.AudioFile("temp_audio.wav") as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
        os.remove("temp_audio.wav")
        return text
    except Exception as e:
        return f"Error processing audio: {e}"

# --- Graph Node Functions ---
def analyze_input_node(state: TutorState):
    initial_input = state['messages'][-1].content
    if "Image Description:" in initial_input or "Audio Transcription:" in initial_input:
        context_summary = initial_input
    else:
        context_summary = f"The user wants to learn about: {initial_input}"
    
    prompt = f"""
    You are an expert educator. Based on the following information, identify the main educational topic and break it down into 3-5 logical, step-by-step sub-topics for teaching.
    Information Provided: {context_summary}
    Respond in the following format, and nothing else:
    Topic: [The main topic]
    Sub-topics:
    1. [First sub-topic]
    2. [Second sub-topic]
    ...
    """
    response = llm.invoke([SystemMessage(content=prompt)])
    response_text = response.content
    lines = response_text.strip().split('\n')
    topic = lines[0].replace("Topic: ", "").strip()
    sub_topics = [line.split('. ', 1)[1] for line in lines if '. ' in line and line.strip().startswith(tuple(f"{i}." for i in range(1, 10)))]
    
    ai_message = AIMessage(content=f"Great! It looks like we're going to learn about **{topic}**. I've prepared a small plan for us:\n\n" + "\n".join([f"**{i+1}.** {s}" for i, s in enumerate(sub_topics)]) + "\n\nReady to start with the first step?")
    
    return {
        "topic": topic,
        "sub_topics": sub_topics,
        "current_step": -1,
        "context_summary": context_summary,
        "messages": state['messages'] + [ai_message]
    }

def explain_step_node(state: TutorState):
    current_step = state['current_step'] + 1
    if current_step >= len(state['sub_topics']):
        ai_message = AIMessage(content="We've covered all the planned topics! Would you like to review anything or ask more specific questions?")
        return {"messages": state['messages'] + [ai_message]}
    
    sub_topic_to_explain = state['sub_topics'][current_step]
    prompt = f"""
    You are an expert and friendly tutor. Your current main topic is '{state['topic']}'.
    Please explain the following sub-topic in a clear, concise, and interactive way. Use analogies and ask a simple question at the end to check for understanding.
    Sub-topic to explain: "{sub_topic_to_explain}"
    Relevant context from the original material: {state['context_summary']}
    """
    response = llm.invoke([SystemMessage(content=prompt)])
    ai_message = AIMessage(content=response.content)
    
    return {"current_step": current_step, "messages": state['messages'] + [ai_message]}

def answer_question_node(state: TutorState):
    user_question = state['user_question']
    prompt = f"""
    You are an expert and friendly tutor. Your current main topic is '{state['topic']}'.
    The user has asked a specific question. Answer it clearly based on the conversation history and the provided context.
    Context from original material: {state['context_summary']}
    Conversation History: {[msg.content for msg in state['messages']]}
    User's Question: "{user_question}"
    Answer the user's question directly.
    """
    response = llm.invoke([SystemMessage(content=prompt)])
    ai_message = AIMessage(content=response.content + "\n\nLet me know if you're ready to continue with our plan or have more questions!")
    return {"messages": state['messages'] + [ai_message]}

# --- Graph Conditional Edges ---
def route_user_input(state: TutorState):
    if "user_question" not in state or state["user_question"] is None:
        return END

    user_message = state['user_question'].lower()
    continue_keywords = ['next', 'continue', 'yes', 'ok', 'sure', 'ready']
    if any(keyword in user_message for keyword in continue_keywords):
        if state['current_step'] >= len(state['sub_topics']) - 1:
            return END
        return "explain_step"
    else:
        return "answer_question"

# --- Build the Graph ---
builder = StateGraph(TutorState)
builder.add_node("analyze_input", analyze_input_node)
builder.add_node("explain_step", explain_step_node)
builder.add_node("answer_question", answer_question_node)
builder.set_entry_point("analyze_input")
builder.add_conditional_edges("analyze_input", lambda x: "explain_step")
builder.add_conditional_edges(
    "explain_step",
    route_user_input,
    {"explain_step": "explain_step", "answer_question": "answer_question", END: END}
)
builder.add_edge("answer_question", END)
tutor_graph = builder.compile()

# --- Streamlit UI ---
st.set_page_config(page_title="Multimodal Educational Tutor", layout="wide")
st.title("🧠 Multimodal Educational Tutor")
st.markdown("Upload a textbook page, a diagram, or record a question to start a lesson!")

if "session_id" not in st.session_state:
    st.session_state.session_id = "tutor_session"
    st.session_state.messages = [SystemMessage(content="Tutor is ready.")]
    st.session_state.graph_state = None

for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("ai").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

uploaded_file = st.sidebar.file_uploader("Upload an image or audio", type=["jpg", "jpeg", "png", "mp3", "wav", "m4a"])
prompt = st.chat_input("Ask a question or say 'next'...")

def start_new_topic(input_content):
    st.session_state.graph_state = None # Reset state for new topic
    st.session_state.messages = [HumanMessage(content=input_content)]
    initial_state = {"messages": st.session_state.messages}
    with st.spinner("Creating a lesson plan..."):
        final_state = tutor_graph.invoke(initial_state, config={"configurable": {"session_id": st.session_state.session_id}})
    st.session_state.graph_state = final_state
    st.session_state.messages = final_state['messages']
    st.rerun()

# --- FILE UPLOAD LOGIC (CORRECTED) ---
if "last_uploaded_file_id" not in st.session_state:
    st.session_state.last_uploaded_file_id = None

if uploaded_file is not None:
    # Use the correct attribute: .file_id
    if uploaded_file.file_id != st.session_state.last_uploaded_file_id:
        # Store the correct attribute: .file_id
        st.session_state.last_uploaded_file_id = uploaded_file.file_id

        start_new_topic_content = None
        file_type = uploaded_file.type.split('/')[0]
        
        if file_type == "image":
            start_new_topic_content = process_image(uploaded_file)
        elif file_type == "audio":
            transcription = process_audio(uploaded_file)
            if transcription.startswith("Error"):
                st.sidebar.error(transcription)
            else:
                start_new_topic_content = f"Audio Transcription: {transcription}"
        
        if start_new_topic_content:
            start_new_topic(start_new_topic_content)

if prompt:
    if not st.session_state.graph_state:
        start_new_topic(prompt)
    else:
        current_state = st.session_state.graph_state
        current_state['messages'].append(HumanMessage(content=prompt))
        current_state['user_question'] = prompt

        with st.spinner("Thinking..."):
            final_state = tutor_graph.invoke(current_state, config={"configurable": {"session_id": st.session_state.session_id}})
        
        st.session_state.graph_state = final_state
        st.session_state.messages = final_state['messages']
        st.rerun()
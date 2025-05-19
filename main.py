import io
import os
import logging
import subprocess
import re
from typing import Any, Callable
from typing_extensions import TypedDict

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import openai
from openai import OpenAI as OpenAIClient
from elevenlabs.client import ElevenLabs
from httpx import ConnectError

from langchain_core.exceptions import OutputParserException
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent
from langchain.agents.mrkl.output_parser import MRKLOutputParser

from langgraph.graph import StateGraph, START, END

from agent_tools import user_points_details, user_ticket_status

# Config and API Keys

load_dotenv()
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
if not (OPENAI_API_KEY and ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID):
    raise RuntimeError("Missing one of OPENAI_API_KEY, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID")

# Set up logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Initialize API clients

oepnai_api = OPENAI_API_KEY
openai.api_key = oepnai_api
audio_client  = OpenAIClient(api_key=OPENAI_API_KEY)
eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Logging wrapper for the tools

def log_tool(func: Callable[[str], str], name: str) -> Callable[[str], str]:
    def wrapper(arg: str) -> str:
        logger.info("Tool %s called with input: %s", name, arg)
        result = func(arg)
        logger.info("Tool %s output: %s", name, result)
        return result
    return wrapper

# Initialize tools

tools = [
    Tool(
        name="get_user_points",
        func=log_tool(user_points_details, "get_user_points"),
        description="Fetch loyalty points for a user number (format U####)."
    ),
    Tool(
        name="get_ticket_status",
        func=log_tool(user_ticket_status, "get_ticket_status"),
        description="Fetch ticket status for a 6-character alphanumeric ID."
    ),
]

# JSON Parse handler

class FenceStrippingParser(MRKLOutputParser):
    def parse(self, text: str):
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        return super().parse(cleaned)

# initializing LLM and Agent

llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2)
system_prompt = (
    "You are a friendly, helpful customer support agent for Concordia Airlines. "
    "You can call get_user_points or get_ticket_status as needed. "
    "Ask clarifying questions if the user is unclear."
)
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=False,
    handle_parsing_errors=True,
    output_parser=FenceStrippingParser(),
    max_iterations=1,
    early_stopping_method="generate",
    agent_kwargs={
        "prefix": system_prompt,
        "suffix": "\nUser: {input}\nAssistant: "
    },
)

# Defining data types

class State(TypedDict, total=False):
    audio_bytes: bytes
    transcription: str
    response_text: str
    tts_stream: Any

# Setting up LangGraph - defining the nodes

async def whisper_node(state: State) -> State:
    audio_data = state.get("audio_bytes")
    if not audio_data:
        raise HTTPException(400, "No audio provided to transcribe")
    logger.info("Starting Whisper transcription")
    try:
        resp = audio_client.audio.transcriptions.create(
            model="whisper-1",
            file=("audio.webm", io.BytesIO(audio_data), "audio/webm")
        )
    except Exception:
        proc = subprocess.Popen(
            ["ffmpeg","-i","pipe:0","-ar","16000","-ac","1","-f","wav","pipe:1"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        wav_bytes, err = proc.communicate(audio_data)
        if proc.returncode != 0:
            logger.error("ffmpeg conversion error: %s", err.decode())
            raise HTTPException(500, "Audio conversion failed")
        resp = audio_client.audio.transcriptions.create(
            model="whisper-1",
            file=("audio.wav", io.BytesIO(wav_bytes), "audio/wav")
        )
    text = getattr(resp, 'text', None)
    if not text:
        raise HTTPException(500, "Empty transcription from Whisper")
    logger.info("Transcribed text: %s", text)
    return {"transcription": text}

async def agent_node(state: State) -> State:
    user_text = state.get("transcription", "")
    logger.info("Invoking agent with transcription: %s", user_text)
    try:
        result = await agent.ainvoke({"input": user_text})
        response = result.get("output") if isinstance(result, dict) else result
    except OutputParserException as e:
        # On parsing errors, return the LLM's raw message
        response = str(e)
        logger.info("Clarification from LLM: %s", response)
    logger.info("LLM generated response: %s", response)
    return {"response_text": response}

async def tts_node(state: State) -> State:
    text = state.get("response_text", "")
    logger.info("Starting TTS synthesis")
    try:
        stream = eleven_client.text_to_speech.convert_as_stream(
            text=text,
            voice_id=ELEVENLABS_VOICE_ID,
            model_id="eleven_turbo_v2_5"
        )
    except ConnectError as e:
        logger.error("TTS connection failed: %s", e)
        raise HTTPException(502, "TTS service unavailable")
    return {"tts_stream": stream}

# Build and compile LangGraph nodes

builder = StateGraph(State)
builder.add_node("whisper", whisper_node)
builder.add_node("agent", agent_node)
builder.add_node("tts", tts_node)
builder.add_edge(START, "whisper")
builder.add_edge("whisper", "agent")
builder.add_edge("agent", "tts")
builder.add_edge("tts", END)
graph = builder.compile()

# Serve the Agent through a fastAPI endpoint

app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])

@app.post("/chat")
async def chat(audio_file: UploadFile = File(...)):
    audio_bytes = await audio_file.read()
    if not audio_bytes:
        raise HTTPException(400, "No audio received")
    try:
        result = await graph.ainvoke({"audio_bytes": audio_bytes})
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        raise HTTPException(500, f"Pipeline error: {e}")
    stream = result.get("tts_stream")
    if not stream:
        raise HTTPException(500, "No TTS audio returned")
    return StreamingResponse(stream, media_type="audio/mpeg")

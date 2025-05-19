# Voice to Voice Agentic AI API

## Summary

This is a Python FastAPI endpoint that uses LangGraph by LangChain, as well as OpenAI Whisper (STT) and ElevenLabs Turbo 2.5 (TTS) to create a simple customer support agent for a fictional airline - Concordia.

The agent has 2 tools at their disposal:

- **user_points_details**: enables the agent to retrieve the user's loyalty points should they provide their loyalty account number
- **user_ticket_status**: enables the agent to retrieve the user's ticket status should they provide the ticket number

It's a simple project. The purpose was to get familiar with agent frameworks as well as sequential voice-to-voice conversational AI with control over the entire codebase rather than platform portals for Vapi and ElevenLabs.

## How to Test

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Set up environment variables**
   - Copy `.env.example` to `.env`
   - Populate the API keys and Voice ID

3. **Set up a virtual environment**
   ```bash
   python -m venv {env_name}
   ```

4. **Activate virtual environment**
   - **Windows:**
     ```bash
     .\{env_name}\Scripts\activate
     ```
   - **Mac/Linux:**
     ```bash
     source ./{env_name}/bin/activate
     ```

5. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

6. **Run the FastAPI using uvicorn**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000 --log-level debug
   ```

7. **Open the web interface**
   - Open `webui.html` in your browser and begin testing

## Technologies Used

- **FastAPI** - Web framework for building APIs
- **LangGraph by LangChain** - Agent framework
- **OpenAI Whisper** - Speech-to-Text (STT)
- **ElevenLabs Turbo 2.5** - Text-to-Speech (TTS)
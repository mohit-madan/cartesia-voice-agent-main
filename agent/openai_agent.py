from livekit.agents.llm import ChatContext, ChatMessage
from livekit.plugins import openai, deepgram, cartesia
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import silero

def create_initial_chat_context() -> ChatContext:
    return ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content="You are a voice assistant created by LiveKit. Your interface with users will be voice. Pretend we're having a conversation, no special formatting or headings, just natural speech.",
            )
        ]
    )

def create_voice_agent(vad, voice_id: str = "248be419-c632-4f23-adf1-5324ed7dbf1d", model: str = "sonic") -> VoicePipelineAgent:
    tts = cartesia.TTS(
        voice=voice_id,
        model=model,
    )
    
    return VoicePipelineAgent(
        vad=vad,
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=tts,
        chat_ctx=create_initial_chat_context(),
    ) 
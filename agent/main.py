import asyncio
import json
import os
import requests

from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli, JobProcess, llm
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
)
from typing import Annotated

from livekit.agents.pipeline import VoicePipelineAgent
from livekit.agents.log import logger
from livekit.plugins import deepgram, silero, cartesia, openai, rag
from typing import List, Any

from dotenv import load_dotenv
from openai_agent import create_voice_agent

import numpy as np
import pickle
import pdb

load_dotenv()

annoy_index = rag.annoy.AnnoyIndex.load('data/wise_faq_vdb')
embeddings_dimension = 1536

with open('./data/wise_faq_documents1.pkl', 'rb') as f:
    faq_data = pickle.load(f)



# Extract embeddings and build lookup
# embeddings = np.array([doc['embedding'] for doc in faq_data]).astype('float32')
# doc_lookup = {i: doc for i, doc in enumerate(faq_data)}

_chat_ctx_lock = asyncio.Lock()

def prewarm(proc: JobProcess):
    # preload models when process starts to speed up first interaction
    proc.userdata["vad"] = silero.VAD.load()

    # fetch cartesia voices

    headers = {
        "X-API-Key": os.getenv("CARTESIA_API_KEY", ""),
        "Cartesia-Version": "2024-08-01",
        "Content-Type": "application/json",
    }
    response = requests.get("https://api.cartesia.ai/voices", headers=headers)
    if response.status_code == 200:
        proc.userdata["cartesia_voices"] = response.json()
    else:
        logger.warning(f"Failed to fetch Cartesia voices: {response.status_code}")

async def _enrich_with_rag(agent: VoicePipelineAgent, chat_ctx: llm.ChatContext) -> None:
    """
    Locate the last user message, use it to query the RAG model for
    the most relevant paragraph, add that to context, and generate a response.
    """
    async with _chat_ctx_lock:
        user_msg = chat_ctx.messages[-1]
    # Let's sleep for 5 seconds to simulate a delay
    await asyncio.sleep(2)
    # user_embedding = model.encode(user_msg.content)
    user_embedding = await openai.create_embeddings(
        input = [user_msg.content],
        model = "text-embedding-3-small",
        dimensions = embeddings_dimension,
    )
    result = annoy_index.query(user_embedding[0].embedding, n=1)[0]
    # print(result)
    result_doc = faq_data[result.userdata]
    # relevant_docs = get_relevant_docs(user_embedding, k=3, distance_threshold=0.0)
    # relevant_docs = annoy_index.get_nns_by_vector(user_embedding, 3, include_distances=True)
    if result_doc:
        logger.info(f"Found {result_doc.get('url', 'No URL found')}")
        rag_msg = llm.ChatMessage.create(
            text = "Context:\n" + result_doc.get('content', ''),
            role = "assistant",
        )
        async with _chat_ctx_lock:
            # Replace last message with RAG, then append user message at the end
            chat_ctx.messages[-1] = rag_msg
            chat_ctx.messages.append(user_msg)

            # Generate a response using the enriched context
            llm_stream = agent._llm.chat(chat_ctx=chat_ctx)
            await agent.say(llm_stream)

def create_initial_chat_context() -> ChatContext:
    return ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content="""You are a helpful Wise customer support agent. Answer questions using only the context provided. 
                If the needed information isn't there, kindly let the customer know and offer to connect them with a human agent. 
                Keep your tone friendly, professional, conversational, and concise.
                When replying from the context, reply in conversational manner, not bullet points. Keep it short and concise.""",
            )
        ]
    )

async def entrypoint(ctx: JobContext):
    cartesia_voices: List[dict[str, Any]] = ctx.proc.userdata["cartesia_voices"]
    fnc_ctx = llm.FunctionContext()
    tts = cartesia.TTS(
        voice="248be419-c632-4f23-adf1-5324ed7dbf1d",
        model="sonic",
    )
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=tts,
        chat_ctx=create_initial_chat_context(),
        fnc_ctx=fnc_ctx,
    ) 

    
    is_user_speaking = False
    is_agent_speaking = False

    @fnc_ctx.ai_callable()
    async def connect_to_human_agent(
        code: Annotated[
            int, llm.TypeInfo(description="Connect to a human agent.")
        ]
    ):
        """
        Called when you need to connect to a human agent.
        """
        logger.info("Connecting to a human agent")
        try:
            # add a speech before disconnecting
            await agent.say("I'm transferring you to a human agent. Please hold on.", allow_interruptions=False)
            await asyncio.sleep(5)
            # Send RPC to all standard participants to end the call
            await ctx.room.local_participant.publish_data(
                "endCall",
                topic="endCall",
            )
            logger.info("Successfully notified frontend to end call")
            return "Successfully notified frontend to end call"
        except Exception as e:
            logger.error(f"Failed to notify frontend: {str(e)}")
            return "Failed to notify frontend"

    # Define the function to enrich with RAG
    @fnc_ctx.ai_callable()
    async def enrich_with_rag(
        code: Annotated[
            int, llm.TypeInfo(description="Enrich with RAG for questions about Wise related to where is my money.")
        ]
    ):
        """
        Called when you need to enrich with RAG for questions about Wise.
        """
        logger.info("Enriching with RAG for questions about Wise")
        async with _chat_ctx_lock:
            thinking_ctx = llm.ChatContext().append(
                role="system",
                text="Generate a very short message to indicate that we're looking up the answer in the docs"
            )
            thinking_stream = agent._llm.chat(chat_ctx=thinking_ctx)
            # Wait for thinking message to complete before proceeding
            await agent.say(thinking_stream, add_to_chat_ctx=False)
        await _enrich_with_rag(agent, agent.chat_ctx)

    @ctx.room.on("participant_attributes_changed")
    def on_participant_attributes_changed(
        changed_attributes: dict[str, str], participant: rtc.Participant
    ):
        # check for attribute changes from the user itself
        if participant.kind != rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD:
            return

        if "voice" in changed_attributes:
            voice_id = participant.attributes.get("voice")
            logger.info(
                f"participant {participant.identity} requested voice change: {voice_id}"
            )
            if not voice_id:
                return

            voice_data = next(
                (voice for voice in cartesia_voices if voice["id"] == voice_id), None
            )
            if not voice_data:
                logger.warning(f"Voice {voice_id} not found")
                return
            if "embedding" in voice_data:
                language = "en"
                if "language" in voice_data and voice_data["language"] != "en":
                    language = voice_data["language"]
                agent.tts._opts.voice = voice_data["embedding"]
                agent.tts._opts.language = language
                # allow user to confirm voice change as long as no one is speaking
                if not (is_agent_speaking or is_user_speaking):
                    asyncio.create_task(
                        agent.say("How do I sound now?", allow_interruptions=True)
                    )

    await ctx.connect()

    @agent.on("agent_started_speaking")
    def agent_started_speaking():
        nonlocal is_agent_speaking
        is_agent_speaking = True

    @agent.on("agent_stopped_speaking")
    def agent_stopped_speaking():
        nonlocal is_agent_speaking
        is_agent_speaking = False

    @agent.on("user_started_speaking")
    def user_started_speaking():
        nonlocal is_user_speaking
        is_user_speaking = True

    @agent.on("user_stopped_speaking")
    def user_stopped_speaking():
        nonlocal is_user_speaking
        is_user_speaking = False

    # set voice listing as attribute for UI
    voices = []
    for voice in cartesia_voices:
        voices.append(
            {
                "id": voice["id"],
                "name": voice["name"],
            }
        )
    voices.sort(key=lambda x: x["name"])
    await ctx.room.local_participant.set_attributes({"voices": json.dumps(voices)})

    agent.start(ctx.room)
    await agent.say("Hi there, I am Claire from Wise. How can I help you today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

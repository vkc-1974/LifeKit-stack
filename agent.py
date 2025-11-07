#!/bin/env python3

# agent.py
import asyncio
import os
import httpx
from livekit import agents
from livekit.agents import Agent, AgentSession, JobContext, function_tool
from livekit.plugins import silero, openai

# === ЛОКАЛЬНЫЕ МОДЕЛИ ===
from faster_whisper import WhisperModel
import edge_tts

MCP_URL = os.getenv("MCP_SERVER_URL", "http://mcp:8000")

# === STT: faster-whisper (ЛОКАЛЬНЫЙ) ===
class LocalSTT:
    def __init__(self):
        self.model = WhisperModel("tiny", device="cpu", compute_type="int8")

    async def transcribe(self, audio):
        segments, _ = self.model.transcribe(audio, language="ru")
        return " ".join([s.text for s in segments])

    @property
    def capabilities(self):
        return agents.STTCapabilities(streaming=False, interim_results=False)

# === TTS: edge-tts (ЛОКАЛЬНЫЙ) ===
async def local_tts(text: str):
    communicate = edge_tts.Communicate(text, "ru-RU-SvetlanaNeural")
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]

    @property
    def capabilities(self):
        return agents.TTSCapabilities(streaming=True)

# === ИНСТРУМЕНТЫ ===
@function_tool
async def get_user_balance(user_id: int) -> str:
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(f"{MCP_URL}/tools/get_user_balance", json={"user_id": user_id})
            return resp.json().get("content", "Ошибка")
        except Exception as e:
            return f"Ошибка: {e}"

@function_tool
async def get_weather(city: str) -> str:
    return f"В городе {city} завтра ясно, температура +25°C"

# === АГЕНТ ===
class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="Ты голосовой помощник. Используй инструменты. Отвечай кратко по-русски.",
            tools=[get_user_balance, get_weather]
        )

# === ENTRYPOINT ===
async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)

    llm = openai.LLM.with_ollama(
        model="llama3.2:3b-instruct-q4_K_M",
        base_url="http://ollama:11434"
    )

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=LocalSTT(),
        llm=llm,
        tts=local_tts,
    )

    await session.start(room=ctx.room, agent=VoiceAgent())
    await session.generate_reply(instructions="Привет! Задавай вопрос о балансе или погоде.")

    await asyncio.Event().wait()

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))

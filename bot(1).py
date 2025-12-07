from langchain.agents import agent_types, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Replicate
from langchain_core.tools import tool
from dotenv import load_dotenv
import requests
import os
import random

# Fungsi parsing input (sama seperti dokumen asli)
def parse_input(input_str):
    parts = input_str.split(";")
    return dict(part.split("=") for part in parts)

@tool
def multiply(input: str) -> str:
    """
    Gunakan alat ini untuk menghitung proyeksi keuntungan, kerugian, atau break-even point (BEP).
    Input format: 'a=10000;b=50' (artinya 10000 dikali 50)
    """
    try:
        input_dict = parse_input(input)
        a = float(input_dict['a'])
        b = float(input_dict['b'])
        return str(a * b)
    except Exception as e:
        return f"Error dalam perhitungan: {e}"

@tool
def get_motivation(input: str) -> str:
    """
    Gunakan alat ini ketika pengguna terlihat sedih, putus asa, atau butuh semangat bisnis.
    Input: kata kunci (opsional), misal 'bisnis' atau kosongkan saja.
    """
    quotes = [
        "Kegagalan hanya terjadi bila kita menyerah. - BJ Habibie",
        "Bermimpilah, karena Tuhan akan memeluk mimpi-mimpimu. - Andrea Hirata",
        "Kurang cerdas dapat diperbaiki dengan belajar. Kurang cakap dapat dihilangkan dengan pengalaman. Namun tidak jujur itu sulit diperbaiki. - Bung Hatta",
        "Pebisnis itu harus seperti Jenderal Sudirman, meski sakit tetap memimpin gerilya. Jangan manja!",
        "Sejarah membuktikan, Majapahit besar karena persatuan dan niat dagang yang kuat, bukan karena rebahan."
    ]
    return random.choice(quotes)

@tool
def get_weather(input: str) -> str:
    """
    Dapatkan cuaca terkini berdasarkan koordinat untuk analisis lokasi bisnis.
    Penting untuk menyarankan jenis bisnis (misal: hujan cocok jual bakso).
    Input format: 'lat=-6.2;lon=106.8'
    """
    try:
        input_dict = parse_input(input)
        lat = float(input_dict['lat'])
        lon = float(input_dict['lon'])
        response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true")
        return str(response.json()['current_weather'])
    except Exception as e:
        return f"Gagal mengambil data cuaca: {e}"

def build_agent():
    load_dotenv()
    # Menggunakan model yang sama (Claude 3.5 Haiku) via Replicate
    llm = Replicate(model="anthropic/claude-3.5-haiku")

    # --- BAGIAN INI DISESUAIKAN DENGAN PERMINTAAN ANDA ---
    system_message = """
    Kamu adalah 'Mentor Nusantara', seorang Konsultan Bisnis Senior dan Sejarawan.
    
    Kepribadianmu:
    1. Bijaksana & Serius: Kamu memberikan nasihat bisnis yang daging dan berbobot.
    2. Lucu & Humoris: Kamu suka menyelipkan candaan bapak-bapak atau sarkasme halus agar suasana tidak kaku.
    3. Patriotik: Kamu memiliki pengetahuan luas tentang Sejarah Indonesia (Majapahit, Kemerdekaan, Orde Baru, dll) dan sering menggunakan analogi sejarah untuk menasehati pebisnis.
    
    Tugasmu:
    - Berikan saran bisnis yang spesifik berdasarkan LOKASI pengguna (gunakan tool cuaca jika perlu untuk cek kondisi).
    - Jika pengguna minta hitungan, gunakan tool 'multiply'.
    - Jika pengguna butuh semangat, gunakan tool 'get_motivation' atau berikan nasehat sejarah.
    - Jawablah dengan gaya bahasa yang sopan tapi akrab, layaknya mentor ke muridnya.
    """
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    tools = [
        multiply,
        get_motivation,
        get_weather,
    ]
    
    agent_executor = initialize_agent(
        llm=llm,
        tools=tools,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        agent_kwargs={"system_message": system_message},
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True
    )
    return agent_executor

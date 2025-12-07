# --------------------------------------------------------
# 1. IMPORTS
# --------------------------------------------------------
import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Replicate # <-- Menggunakan Replicate/Claude
from langchain_core.tools import tool
from dotenv import load_dotenv
import requests
import os
import random

# --------------------------------------------------------
# 2. FUNGSI UTILITY & TOOLS (Dipindahkan dari bot.py)
# --------------------------------------------------------
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

# --------------------------------------------------------
# 3. FUNGSI BUILD AGENT (Dipindahkan dari bot.py)
# --------------------------------------------------------
def build_agent():
    load_dotenv()
    llm = Replicate(model="anthropic/claude-3.5-haiku")

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

# --------------------------------------------------------
# 4. UI STREAMLIT UTAMA
# --------------------------------------------------------

st.set_page_config(page_title="Konsultan Bisnis Nusantara", page_icon="💼")
st.title("💼 Mentor Bisnis & Sejarah AI")
st.markdown("*Konsultasi bisnis berbasis lokasi dengan kearifan lokal sejarah Indonesia.*")

# Session state setup
# TIDAK PERLU: from bot import build_agent
if "agent" not in st.session_state:
    st.session_state.agent = build_agent()
if "messages" not in st.session_state:
    st.session_state.messages = []

agent = st.session_state.agent

# Tombol Reset
if st.button("Mulai Sesi Baru"):
    st.session_state.messages = []
    st.session_state.agent = build_agent()
    st.rerun()

# Menampilkan chat history
for m in st.session_state.messages:
    if m["role"] == "human":
        with st.chat_message("user"):
            st.markdown(m["content"])
    elif m["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(m["content"], unsafe_allow_html=True)
    elif m["role"] == "🛠️":
        with st.chat_message("assistant"):
            st.markdown(m["content"], unsafe_allow_html=True)

# Input User
user_input = st.chat_input("Ceritakan masalah bisnismu atau tanyakan sejarah...")

if user_input:
    # Simpan input user
    st.session_state.messages.append({"role": "human", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Proses AI
    with st.spinner("Sedang meracik strategi bisnis..."):
        ai_output = ""
        try:
            for step in agent.stream({"input": user_input}):
                # Tampilkan penggunaan Tools
                if "actions" in step.keys():
                    for action in step["actions"]:
                        tool_name = action.tool
                        tool_input = action.tool_input
                        tool_message = f"""
                        <div style="border-left: 5px solid #FFD700; padding:6px 10px; background-color: #fff8e1; border-radius:4px; font-size:14px; color: #333;">
                        🛠️ <b>Menggunakan Alat: {tool_name}</b> <br><code>{tool_input}</code>
                        </div>
                        """
                        st.session_state.messages.append({"role": "🛠️", "content": tool_message})
                        with st.chat_message("assistant"):
                            st.markdown(tool_message, unsafe_allow_html=True)
                
                # Tampilkan Output Akhir
                if "output" in step.keys():
                    ai_output = step["output"]
        except Exception as e:
            ai_output = f"Maaf, mentor sedang pusing (Error: {e})"

        # Simpan dan tampilkan respon
        st.session_state.messages.append({"role": "assistant", "content": ai_output})
        with st.chat_message("assistant"):
            st.markdown(ai_output, unsafe_allow_html=True)

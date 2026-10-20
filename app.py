import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

st.title("🤖 Analista IA - Equipo SST")
st.write("Pregúntame cualquier dato sobre el dashboard de Bukplay.")

# 1. Traer la clave mágica de Gemini desde los "Secretos"
os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

# 2. Conectar a Google Sheets usando tu archivo JSON
@st.cache_data(ttl=3600) # Esto guarda los datos por 1 hora para que el bot responda más rápido
def cargar_datos():
    # Leemos el archivo JSON oculto
    cred_dict = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    credentials = Credentials.from_service_account_info(cred_dict, scopes=scopes)
    gc = gspread.authorize(credentials)
    
    # Abrimos tu planilla y pestaña
    ID_SHEET = "1B2CMaXQ_yNOr4NQROCrv_iiCAxBxhLPcJHy7tBQTD7g" 
    hoja = gc.open_by_key(ID_SHEET).worksheet("Visualizaciones")
    df = pd.DataFrame(hoja.get_all_records())
    
    # Filtro de Privacidad: Solo SST
    if 'Categoría' in df.columns:
        df = df[df['Categoría'].str.contains('SST|Seguridad', case=False, na=False)]
    return df

with st.spinner("Conectando con la base de datos de Looker Studio..."):
    df_sst = cargar_datos()

# 3. Inicializar el cerebro de la IA
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
agente = create_pandas_dataframe_agent(
    llm, df_sst, verbose=False, 
    allow_dangerous_code=True, handle_parsing_errors=True
)

# 4. Construir la interfaz del Chat (Estilo WhatsApp)
if "mensajes" not in st.session_state:
    st.session_state.mensajes = [{"role": "assistant", "content": "¡Hola! Ya leí la base de datos de SST. ¿Qué te gustaría saber hoy?"}]

# Mostrar el historial
for msg in st.session_state.mensajes:
    st.chat_message(msg["role"]).write(msg["content"])

# Cuadro para escribir
pregunta = st.chat_input("Ej: ¿Cuáles son los 3 cursos más vistos?")
if pregunta:
    st.chat_message("user").write(pregunta)
    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    
    with st.spinner("Analizando miles de filas..."):
        try:
            respuesta = agente.invoke(pregunta)
            texto_respuesta = respuesta["output"]
        except Exception as e:
            texto_respuesta = "Uy, me enredé con esos datos. ¿Podrías preguntar de otra forma?"
            
    st.chat_message("assistant").write(texto_respuesta)
    st.session_state.mensajes.append({"role": "assistant", "content": texto_respuesta})

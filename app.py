import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

st.title("🤖 Analista IA - Equipo SST")
st.write("Pregúntame cualquier dato sobre el dashboard de Bukplay.")

# 1. Clave de Gemini
import os
os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

# 2. Cargar datos directamente
@st.cache_data(ttl=3600)
def cargar_datos():
    url = "https://docs.google.com/spreadsheets/d/1B2CMaXQ_yNOr4NQROCrv_iiCAxBxhLPcJHy7tBQTD7g/gviz/tq?tqx=out:csv&sheet=Visualizaciones"
    df = pd.read_csv(url)
    if 'Categoría' in df.columns:
        df = df[df['Categoría'].str.contains('SST|Seguridad', case=False, na=False)]
    return df

with st.spinner("Conectando con la base de datos..."):
    df_sst = cargar_datos()

# 3. Inicializar el cerebro de la IA
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

# Instrucción estricta para evitar errores de parsing
PREFIX = """
Eres un analista experto en datos de SST. 
SIEMPRE debes responder en español y usando el formato ReAct.
Cuando tengas la respuesta final, DEBES escribir 'Final Answer:' antes de tu respuesta.
"""

agente = create_pandas_dataframe_agent(
    llm, 
    df_sst, 
    verbose=False, 
    allow_dangerous_code=True, 
    handle_parsing_errors=True,
    prefix=PREFIX
)

# 4. Historial del Chat
if "mensajes" not in st.session_state:
    st.session_state.mensajes = [{"role": "assistant", "content": "¡Hola! Ya leí la base de datos de SST. ¿Qué te gustaría saber hoy?"}]

for msg in st.session_state.mensajes:
    st.chat_message(msg["role"]).write(msg["content"])

# 5. Procesar Pregunta
pregunta = st.chat_input("Ej: ¿Cuáles son los 3 cursos más vistos?")
if pregunta:
    st.chat_message("user").write(pregunta)
    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    
    with st.spinner("Analizando datos..."):
        try:
            respuesta = agente.invoke(pregunta)
            texto_respuesta = respuesta["output"]
        except Exception as e:
            # Si falla, mostramos el error exacto para saber qué corregir
            texto_respuesta = f"Error técnico: {str(e)}"
            
    st.chat_message("assistant").write(texto_respuesta)
    st.session_state.mensajes.append({"role": "assistant", "content": texto_respuesta})

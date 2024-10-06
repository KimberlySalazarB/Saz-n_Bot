import pandas as pd
import streamlit as st
from datetime import datetime
from copy import deepcopy
from groq import Groq
import csv
import re
import pytz
import json
import logging
# Configura el logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Inicializar el cliente de Groq con la clave API
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Configuración inicial de la página
st.set_page_config(page_title="SazónBot", page_icon=":pot_of_food:")
st.title("🍲 SazónBot")

# Mensaje de bienvenida
intro = """¡Bienvenido a Sazón Bot, el lugar donde todos tus antojos de almuerzo se hacen realidad!
Comienza a chatear con Sazón Bot y descubre qué puedes pedir, cuánto cuesta y cómo realizar tu pago. ¡Estamos aquí para ayudarte a disfrutar del mejor almuerzo!"""
st.markdown(intro)


# Cargar el menú desde un archivo CSV
def load_menu(file_path):
    """Cargar el menú desde un archivo CSV con columnas Plato, Descripción y Precio."""
    menu = pd.read_csv(file_path)
    return menu

# Cargar los distritos de reparto desde un archivo CSV
def load_distritos(file_path):
    """Cargar los distritos de reparto desde un archivo CSV."""
    distritos = pd.read_csv(file_path)
    return distritos

def format_menu(menu):
    if menu.empty:
        return "No hay platos disponibles."

    else:
        # Encabezados de la tabla
        table = "| **Plato** | **Descripción** | **Precio** |\n"
        table += "|-----------|-----------------|-------------|\n"  # Línea de separación
        
        # Filas de la tabla
        for idx, row in menu.iterrows():
            table += f"| {row['Plato']} | {row['Descripción']} | S/{row['Precio']:.2f} |\n"
        
        return table


# Mostrar el menú con descripciones
def display_menu(menu):
    """Mostrar el menú con descripciones."""
    menu_text = "Aquí está nuestra carta:\n"
    for index, row in menu.iterrows():
        menu_text += f"{row['Plato']}: {row['Descripción']} - {row['Precio']} soles\n"
    return menu_text

# Mostrar los distritos de reparto
def display_distritos(distritos):
    """Mostrar los distritos de reparto disponibles."""
    distritos_text = "Los distritos de reparto son:\n"
    for index, row in distritos.iterrows():
        distritos_text += f"**{row['Distrito']}**\n"
    return distritos_text

# Cargar el menú y distritos
menu = load_menu("carta.csv")
distritos = load_distritos("distritos.csv")

def display_confirmed_order(order_details):
    """Genera una tabla en formato Markdown para el pedido confirmado."""
    table = "| **Plato** | **Cantidad** | **Precio Total** |\n"
    table += "|-----------|--------------|------------------|\n"
    for item in order_details:
        table += f"| {item['Plato']} | {item['Cantidad']} | S/{item['Precio Total']:.2f} |\n"
    table += "| **Total** |              | **S/ {:.2f}**      |\n".format(sum(item['Precio Total'] for item in order_details))
    return table

##Pendiente


def get_system_prompt(menu, distritos):
    """Definir el prompt del sistema para el bot de Sazón incluyendo el menú y distritos."""
    lima_tz = pytz.timezone('America/Lima') # Define la zona horaria de Lima
    hora_lima = datetime.now(lima_tz).strftime("%Y-%m-%d %H:%M:%S") # Obtiene la hora actual en Lima
    system_prompt = f"""
    Eres el bot de pedidos de Sazón. Ayudas a los clientes a hacer sus pedidos y siempre 
    eres bien amable. Aquí tienes el menú para que se lo muestres a los clientes:\n{display_menu(menu)}\n
    También repartimos en los siguientes distritos: {display_distritos(distritos)}.\n
    Primero, saluda al cliente y ofrécele el menú. Luego, pregunta si quiere recoger su pedido en el local o si prefiere que lo enviemos a domicilio. 
    Asegúrate de usar solo español peruano en tus respuestas, evitando cualquier término como preferís debe ser prefiere. 
    Verifica que el cliente haya ingresado el método de pedido antes de continuar. Si el pedido es para entrega, 
    asegúrate de que el distrito esté disponible y confirma con el cliente el distrito de entrega. 
    Si el pedido es para recoger, pregunta si desea recoger en el local. Después, resume 
    el pedido en la siguiente tabla:\n
    | **Plato** | **Cantidad** | **Precio Total** |\n
    |-----------|--------------|------------------|\n
    |           |              |                  |\n
    | **Total** |              | **S/ 0.00**      |\n
    El monto total del pedido no acepta descuentos ni rectificaciones del precio. 

    Pregunta al cliente: "¿Estás de acuerdo con el pedido?" y espera su respuesta. 
    Una vez que confirme, pregunta: "¿Cuál es tu método de pago? ¿Deseas pagar con tarjeta de crédito, efectivo o algún otro método?". 

    Una vez que el cliente confirme el pedido, registra la hora actual de Perú como el timestamp {hora_lima} de la confirmación. 
    El pedido confirmado será:\n
    {display_confirmed_order([{'Plato': '', 'Cantidad': 0, 'Precio Total': 0}])}\n
    Recuerda verificar que el pedido sea correcto antes de registrarlo.
    """
    return system_prompt.replace("\n", " ")
    
def extract_order_json(response):
    """Extrae el pedido confirmado en formato JSON desde la respuesta del bot solo si todos los campos tienen valores completos."""
    prompt = f"Extrae la información del pedido de la siguiente respuesta: '{response}'. Si el pedido está confirmado proporciona una salida en formato JSON con las claves: Platos, Precio total, Método de pago y timestamp_confirmacion. Si el pedido no está confirmado devuelve una lista vacía."
    
    
    extraction = client.chat.completions.create(
        messages=[{"role": "system", "content": "Eres un asistente que solo responde en JSON. Responde únicamente con un JSON o una lista vacía.."},
                  {"role": "user", "content": prompt}],
        model="llama3-8b-8192",
        temperature=0,
        max_tokens=300,
        top_p=1,
        stop=None,
        stream=False,
    )

    response_content = extraction.choices[0].message.content
    # Intenta cargar como JSON
    return response_content



def generate_response(prompt, temperature=0,max_tokens=1000):
    """Enviar el prompt a Groq y devolver la respuesta con un límite de tokens."""
    st.session_state["messages"].append({"role": "user", "content": prompt})

    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=st.session_state["messages"],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    response = completion.choices[0].message.content
    st.session_state["messages"].append({"role": "assistant", "content": response})
    # Extraer JSON del pedido confirmado
    order_json = extract_order_json(response)
    st.markdown(order_json)
    # Registrar en log en formato JSON puro
    logging.info(json.dumps(order_json, indent=4) if order_json else '{}')
    return response

# Ajustar el tono del bot
def adjust_tone(tone="friendly"):
    """Ajustar el tono del bot según las preferencias del cliente."""
    if tone == "formal":
        st.session_state["tone"] = "formal"
        return "Eres un asistente formal y educado."
    else:
        st.session_state["tone"] = "friendly"
        return "Eres un asistente amigable y relajado."

        
initial_state = [
    {"role": "system", "content": get_system_prompt(menu,distritos)},
    {
        "role": "assistant",
        "content": f"¿Qué te puedo ofrecer?\n\nEste es el menú del día:\n\n{format_menu(menu)}",
    },
]


if "messages" not in st.session_state:
    st.session_state["messages"] = deepcopy(initial_state)

# eliminar conversación
clear_button = st.button("Eliminar conversación", key="clear")
if clear_button:
    st.session_state["messages"] = deepcopy(initial_state)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    elif message["role"] == "assistant":
        with st.chat_message(message["role"], avatar="👨‍🍳"):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"], avatar="👤"):
            st.markdown(message["content"])

if prompt := st.chat_input():
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    output = generate_response(prompt)
    with st.chat_message("assistant", avatar="👨‍🍳"):
        st.markdown(output)
    



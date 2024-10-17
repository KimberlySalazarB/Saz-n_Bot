import pandas as pd
import streamlit as st
from datetime import datetime
from copy import deepcopy
from openai import OpenAI
import csv
import re
import pytz
import json
import logging
from word2number import w2n  # Importación necesaria para convertir palabras a números
import unicodedata

# Configura el logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Inicializar el cliente de OpenAI con la clave API
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Configuración inicial de la página
st.set_page_config(page_title="SazónBot", page_icon=":pot_of_food:")
st.title("🍲 SazónBot")

# Mensaje de bienvenida
intro = """¡Bienvenido a Sazón Bot, el lugar donde todos tus antojos de almuerzo se hacen realidad!
Comienza a chatear con Sazón Bot y descubre qué puedes pedir, cuánto cuesta y cómo realizar tu pago. ¡Estamos aquí para ayudarte a disfrutar del mejor almuerzo!"""
st.markdown(intro)

# Cargar el menú desde un archivo CSV
def load(file_path):
    """Cargar el menú desde un archivo CSV con columnas Plato, Descripción y Precio."""
    data = pd.read_csv(file_path)
    return data

def format_menu(menu):
    if menu.empty:
        return "No hay platos disponibles."
    else:
        # Encabezados de la tabla
        table = "| *Plato* | *Descripción* | *Precio* |\n"
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
        distritos_text += f"*{row['Distrito']}*\n"
    return distritos_text

def display_postre(postre):
    """Mostrar el menú con descripciones."""
    postre_text = "Aquí está lista de postres:\n"
    for index, row in postre.iterrows():
        postre_text += f"{row['Postres']}: {row['Descripción']} - {row['Precio']} soles\n"
    return postre_text

def display_bebida(bebida):
    """Mostrar el menú con descripciones."""
    bebida_text = "Aquí está lista de bebidas:\n"
    for index, row in bebida.iterrows():
        bebida_text += f"{row['bebida']}: {row['descripcion']} - {row['precio']} soles\n"
    return bebida_text

# Cargar el menú y distritos
menu = load("carta.csv")
distritos = load("distritos.csv")
bebidas = load("Bebidas.csv")
postres = load("Postres.csv")

# Obtener todos los nombres de productos
menu_items = menu['Plato'].tolist()
bebida_items = bebidas['bebida'].tolist()
postre_items = postres['Postres'].tolist()

# Combinar todos los ítems
all_menu_items = menu_items + bebida_items + postre_items

def display_confirmed_order(order_details):
    """Genera una tabla en formato Markdown para el pedido confirmado."""
    table = "| *Plato* | *Cantidad* | *Precio Total* |\n"
    table += "|-----------|--------------|------------------|\n"
    for item in order_details:
        table += f"| {item['Plato']} | {item['Cantidad']} | S/{item['Precio Total']:.2f} |\n"
    table += "| *Total* |              | *S/ {:.2f}*      |\n".format(sum(item['Precio Total'] for item in order_details))
    return table

def normalize_text(text):
    """Elimina acentos y convierte a minúsculas."""
    text = text.lower()
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    return text

def get_system_prompt(menu, distritos):
    """Define el prompt del sistema para el bot de Sazón incluyendo el menú y distritos."""
    lima_tz = pytz.timezone('America/Lima')  # Define la zona horaria de Lima
    hora_lima = datetime.now(lima_tz).strftime("%Y-%m-%d %H:%M:%S")  # Obtiene la hora actual en Lima
    system_prompt = f"""
    Eres el bot de pedidos de Sazón, amable y servicial. Ayudas a los clientes a hacer sus pedidos y siempre confirmas que solo pidan platos que están en el menú oficial. Aquí tienes el menú para mostrárselo a los clientes:\n{display_menu(menu)}\n
    También repartimos en los siguientes distritos: {display_distritos(distritos)}.\n
    Primero, saluda al cliente y ofrécele el menú. Asegúrate de que el cliente solo seleccione platos que están en el menú actual y explícales que no podemos preparar platos fuera del menú.

    *Interpretación de cantidades:*
    - El cliente puede indicar la cantidad en texto o en números.
    - Convierte cualquier cantidad escrita en palabras a su valor numérico antes de procesarla (por ejemplo, "dieciséis" a 16, "cincuenta" a 50).

    *IMPORTANTE: Validación de cantidad solicitada*
    - Si la cantidad solicitada está en el rango de 1 a 100 (inclusive), acepta el pedido sin mostrar advertencias.
    - Si la cantidad solicitada es mayor que 100, muestra el siguiente mensaje:
      "Lamento informarte que el límite máximo de cantidad por producto es de 100 unidades. Por favor, reduce la cantidad para procesar tu pedido."

    - Si el usuario solicita múltiples productos en un solo mensaje, procesa cada uno de ellos siguiendo las mismas reglas.

    Si el cliente solicita un producto que no está en el menú, infórmale amablemente que no lo tenemos disponible y sugiérele elegir otro plato del menú.

    Pregunta si desea recoger su pedido en el local o si prefiere entrega a domicilio. 
    Si elige entrega, pregúntale al cliente a qué distrito desea que se le envíe su pedido, confirma que el distrito esté dentro de las zonas de reparto y verifica el distrito de entrega con el cliente.
    Si el pedido es para recoger, invítalo a acercarse a nuestro local ubicado en UPCH123.

    Usa solo español peruano en tus respuestas, evitando palabras como "preferís" y empleando "prefiere" en su lugar.

    Antes de continuar, confirma que el cliente haya ingresado un método de entrega válido. Luego, resume el pedido en la siguiente tabla:\n
    | *Plato*      | *Cantidad* | *Precio Total* |\n
    |----------------|--------------|------------------|\n
    |                |              |                  |\n
    | *Total*      |              | *S/ 0.00*      |\n

    Aclara que el monto total del pedido no acepta descuentos ni ajustes de precio.

    Pregunta al cliente si quiere añadir una bebida o postre. 
    - Si responde bebida, muéstrale únicamente la carta de bebidas {display_bebida(bebidas)}.
    - Si responde postre, muéstrale solo la carta de postres {display_postre(postres)}.

    Si el cliente agrega postres o bebidas, incorpóralos en la tabla de resumen como un plato adicional y calcula el monto total nuevamente con precisión.

    Al final, pregúntale al cliente: "¿Estás de acuerdo con el pedido?" y espera su confirmación. 

    Luego de confirmar, pide el método de pago (tarjeta de crédito, efectivo u otra opción disponible). Verifica que haya ingresado un método de pago antes de continuar.

    Una vez que el cliente confirme el método de pago, registra la hora actual de Perú como el timestamp {hora_lima} de la confirmación. 
    El pedido confirmado será:\n
    {display_confirmed_order([{'Plato': '', 'Cantidad': 0, 'Precio Total': 0}])}\n

    Recuerda siempre confirmar que el pedido, el método de pago y el lugar de entrega estén completos y correctos antes de registrarlo.
    """
    return system_prompt.replace("\n", " ")

def extract_order_json(response):
    # ... (sin cambios)
    pass  # Mantén la implementación existente

def generate_response(prompt, temperature=0.5, max_tokens=1000):
    # ... (sin cambios)
    pass  # Mantén la implementación existente

def extract_quantities_and_items(user_input, menu_items):
    user_input = user_input.lower()
    pattern = r'(\d+|\w+)\s+([a-záéíóúñ]+(?:\s+[a-záéíóúñ]+)*)'
    matches = re.findall(pattern, user_input)
    
    quantities = []
    items = []
    
    for quantity_str, item in matches:
        item = item.strip()
        item_normalized = normalize_text(item)
        try:
            # Intentar convertir la cantidad a número
            quantity = int(quantity_str)
        except ValueError:
            try:
                quantity = w2n.word_to_num(quantity_str)
            except ValueError:
                quantity = None
        if quantity is not None and item != '':
            # Verificar si el producto está en el menú
            item_in_menu = False
            for menu_item in menu_items:
                menu_item_normalized = normalize_text(menu_item)
                if menu_item_normalized in item_normalized or item_normalized in menu_item_normalized:
                    item_in_menu = True
                    item = menu_item  # Usar el nombre oficial del menú
                    break
            if item_in_menu:
                quantities.append(quantity)
                items.append(item)
            else:
                # Producto no encontrado en el menú
                pass  # Podrías manejar productos no encontrados si lo deseas
    return quantities, items

initial_state = [
    {"role": "system", "content": get_system_prompt(menu, distritos)},
    {
        "role": "assistant",
        "content": f"¿Qué te puedo ofrecer?\n\nEste es el menú del día:\n\n{format_menu(menu)}",
    },
]

if "messages" not in st.session_state:
    st.session_state["messages"] = deepcopy(initial_state)

# Botón para eliminar conversación
clear_button = st.button("Eliminar conversación", key="clear")
if clear_button:
    st.session_state["messages"] = deepcopy(initial_state)

# Mostrar mensajes del historial
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
    # Extraer cantidades e ítems
    quantities, items = extract_quantities_and_items(prompt, all_menu_items)
    invalid_items = []
    for qty in quantities:
        if qty > 100:
            invalid_items.append(qty)
    
    if invalid_items:
        with st.chat_message("assistant", avatar="👨‍🍳"):
            st.markdown(
                "Lamento informarte que el límite máximo de cantidad por producto es de 100 unidades. Por favor, reduce la cantidad para procesar tu pedido."
            )
    elif not items:
        with st.chat_message("assistant", avatar="👨‍🍳"):
            st.markdown(
                "Lo siento, algunos de los productos que solicitaste no están en nuestro menú. Por favor, revisa el menú y vuelve a intentarlo."
            )
    else:
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        output = generate_response(prompt)
        with st.chat_message("assistant", avatar="👨‍🍳"):
            st.markdown(output)

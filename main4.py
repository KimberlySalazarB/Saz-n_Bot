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
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk

# Descargar datos necesarios de nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

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

def display_confirmed_order(order_details):
    """Genera una tabla en formato Markdown para el pedido confirmado."""
    table = "| **Plato** | **Cantidad** | **Precio Total** |\n"
    table += "|-----------|--------------|------------------|\n"
    for item in order_details:
        table += f"| {item['Plato']} | {item['Cantidad']} | S/{item['Precio Total']:.2f} |\n"
    table += "| **Total** |              | **S/ {:.2f}**      |\n".format(sum(item['Precio Total'] for item in order_details))
    return table

def get_system_prompt(menu, distritos):
    """Define el prompt del sistema para el bot de Sazón incluyendo el menú y distritos."""
    lima_tz = pytz.timezone('America/Lima')  # Define la zona horaria de Lima
    hora_lima = datetime.now(lima_tz).strftime("%Y-%m-%d %H:%M:%S")  # Obtiene la hora actual en Lima
    system_prompt = f"""
    Eres el bot de pedidos de Sazón, amable y servicial. Ayudas a los clientes a hacer sus pedidos y siempre confirmas que solo pidan platos que están en el menú oficial. Aquí tienes el menú para mostrárselo a los clientes:\n{display_menu(menu)}\n
    También repartimos en los siguientes distritos: {display_distritos(distritos)}.\n
    Primero, saluda al cliente y ofrécele el menú. Asegúrate de que el cliente solo seleccione platos que están en el menú actual y explícales que no podemos preparar platos fuera del menú.

    **Interpretación de cantidades:**
    - El cliente puede indicar la cantidad en texto o en números.
    - Convierte cualquier cantidad escrita en palabras a su valor numérico antes de procesarla (por ejemplo, "dieciséis" a 16, "cincuenta" a 50).

    **IMPORTANTE: Validación de cantidad solicitada**
    - Si la cantidad solicitada está en el rango de 1 a 100 (inclusive), acepta el pedido sin mostrar advertencias.
    - Si la cantidad solicitada es mayor que 100, muestra el siguiente mensaje:
      "Lamento informarte que el límite máximo de cantidad por producto es de 100 unidades. Por favor, reduce la cantidad para procesar tu pedido."

    - Si el usuario solicita múltiples productos en un solo mensaje, procesa cada uno de ellos siguiendo las mismas reglas.

    Pregunta si desea recoger su pedido en el local o si prefiere entrega a domicilio. 
    Si elige entrega, pregúntale al cliente a qué distrito desea que se le envíe su pedido, confirma que el distrito esté dentro de las zonas de reparto y verifica el distrito de entrega con el cliente.
    Si el pedido es para recoger, invítalo a acercarse a nuestro local ubicado en UPCH123.

    Usa solo español peruano en tus respuestas, evitando palabras como "preferís" y empleando "prefiere" en su lugar.

    Antes de continuar, confirma que el cliente haya ingresado un método de entrega válido. Luego, resume el pedido en la siguiente tabla:\n
    | **Plato**      | **Cantidad** | **Precio Total** |\n
    |----------------|--------------|------------------|\n
    |                |              |                  |\n
    | **Total**      |              | **S/ 0.00**      |\n

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
    """Extrae el pedido confirmado en formato JSON desde la respuesta del bot solo si todos los campos tienen valores completos."""
    prompt = f"""
    	Extrae únicamente la información visible y explícita del pedido confirmado de la siguiente respuesta: '{response}'.
    	Si el pedido está confirmado en el texto, devuelve el resultado en formato JSON con las siguientes claves:
    	- 'Platos': una lista de platos donde cada plato incluye su cantidad y precio_total.
    	- 'Total': el monto total del pedido.
    	- 'metodo de pago': el método de pago elegido por el cliente.
    	- 'lugar_entrega': el lugar de entrega especificado por el cliente (en el local o en el distrito indicado).
    	- 'timestamp_confirmacion': la marca de tiempo del momento en que se confirma el pedido.

    	Si algún campo como 'metodo de pago', 'lugar_entrega' o 'timestamp_confirmacion' no aparece explícitamente en la respuesta del cliente, asigna el valor null a ese campo.

    	Si el pedido no está confirmado explícitamente en la respuesta, devuelve un diccionario vacío.
    	No generes, interpretes, ni asumas valores que no estén presentes en la respuesta."""
    extraction = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Eres un asistente que extrae el pedido confirmado en JSON. Responde solo con un JSON o un diccionario vacío."},
            {"role": "user", "content": prompt}
        ],
        model="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=300,
        top_p=1,
        stop=None,
        stream=False,
    )
    response_content = extraction.choices[0].message.content

    # Intenta cargar como JSON
    try:
        order_json = json.loads(response_content)
        st.markdown(order_json)
        st.markdown(type(order_json))
        # Verifica si el JSON es un diccionario
        if isinstance(order_json, dict):
            if all(order_json[key] not in (None, '', [], {}) for key in order_json):
                return order_json
            else:
                print("Advertencia: Hay claves con valores nulos o vacíos en el pedido.")
                return {}
        elif isinstance(order_json, list):
            print("Advertencia: Se recibió una lista en lugar de un diccionario.")
            return {}
        else:
            return {}
    except json.JSONDecodeError:
        # Manejo de error en caso de que el JSON no sea válido
        return {}

def generate_response(prompt, temperature=0.5, max_tokens=1000):
    """Enviar el prompt a OpenAI y devolver la respuesta con un límite de tokens."""
    st.session_state["messages"].append({"role": "user", "content": prompt})

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
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
    st.markdown(type(order_json))
    logging.info(json.dumps(order_json, indent=4) if order_json else '{}')
    return response

# Función para convertir cantidades escritas en palabras a números y extraer productos
def extract_quantities_and_items(user_input):
    user_input = user_input.lower()
    tokens = word_tokenize(user_input)
    tagged_tokens = pos_tag(tokens)

    quantities = []
    items = []
    quantity = None
    item = ''

    for word, tag in tagged_tokens:
        if tag == 'CD':  # Cardinal numbers
            try:
                # Intenta convertir el número en texto a entero
                quantity = w2n.word_to_num(word)
            except ValueError:
                if word.isdigit():
                    quantity = int(word)
                else:
                    quantity = None
        elif tag == 'NN' or tag == 'NNS' or tag == 'JJ':
            item += word + ' '
        elif word == ',' or word == 'y':
            if quantity is not None and item.strip() != '':
                quantities.append(quantity)
                items.append(item.strip())
                quantity = None
                item = ''
        else:
            item += word + ' '

    if quantity is not None and item.strip() != '':
        quantities.append(quantity)
        items.append(item.strip())

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
    quantities, items = extract_quantities_and_items(prompt)
    invalid_items = []
    for qty in quantities:
        if qty > 100:
            invalid_items.append(qty)

    if invalid_items:
        with st.chat_message("assistant", avatar="👨‍🍳"):
            st.markdown(
                "Lamento informarte que el límite máximo de cantidad por producto es de 100 unidades. Por favor, reduce la cantidad para procesar tu pedido."
            )
    else:
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        output = generate_response(prompt)
        with st.chat_message("assistant", avatar="👨‍🍳"):
            st.markdown(output)

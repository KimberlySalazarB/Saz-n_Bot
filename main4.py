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
    load = pd.read_csv(file_path)
    return load

# Funciones para mostrar el menú, bebidas, postres y distritos
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

def display_menu(menu):
    """Mostrar el menú con descripciones."""
    menu_text = "Aquí está nuestra carta:\n"
    for index, row in menu.iterrows():
        menu_text += f"{row['Plato']}: {row['Descripción']} - {row['Precio']} soles\n"
    return menu_text

def display_distritos(distritos):
    """Mostrar los distritos de reparto disponibles."""
    distritos_text = "Los distritos de reparto son:\n"
    for index, row in distritos.iterrows():
        distritos_text += f"**{row['Distrito']}**\n"
    return distritos_text

def display_postre(postre):
    """Mostrar el menú en formato de tabla."""
    # Encabezado de la tabla
    menu_text = "Aquí está nuestra carta de postres:\n"
    menu_text += "| Postre           | Descripción                 | Precio (S/) |\n"
    menu_text += "|------------------|-----------------------------|-------------|\n"
    
    # Agregar cada postre a la tabla
    for index, row in postre.iterrows():
        menu_text += f"| {row['Postres']:<18} | {row['Descripción']:<27} | {row['Precio']:>11} |\n"
    
    return menu_text

def display_bebida(bebida):
    """Mostrar el menú en formato de tabla."""
    # Encabezado de la tabla
    menu_text = "Aquí está nuestra carta de bebidas:\n"
    menu_text += "| Bebida           | Descripción                 | Precio (S/) |\n"
    menu_text += "|------------------|-----------------------------|-------------|\n"
    
    # Agregar cada bebida a la tabla
    for index, row in bebida.iterrows():
        menu_text += f"| {row['bebida']:<18} | {row['descripcion']:<27} | {row['precio']:>11} |\n"
    
    return menu_text

def display_confirmed_order(order_details):
    """Genera una tabla en formato Markdown para el pedido confirmado."""
    table = "| **Plato** | **Cantidad** | **Precio Total** |\n"
    table += "|-----------|--------------|------------------|\n"
    for item in order_details:
        table += f"| {item['Plato']} | {item['Cantidad']} | S/{item['Precio Total']:.2f} |\n"
    table += "| **Total** |              | **S/ {:.2f}**      |\n".format(sum(item['Precio Total'] for item in order_details))
    return table

# Funciones para convertir palabras a números y verificar el rango
def palabras_a_numero(palabra):
    """Convierte un número en palabras o cifras a su valor numérico."""
    numeros = {
        # Unidades
        "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
        "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
        # Decenas
        "diez": 10, "once":11, "doce":12, "trece":13, "catorce":14, "quince":15,
        "dieciséis":16, "diecisiete":17, "dieciocho":18, "diecinueve":19, "veinte": 20, "veintiuno":21, "veintidós":22, "veintitrés":23, "veinticuatro":24, "veinticinco":25, "veintiséis":26, "veintisiete":27, "veintiocho":28, "veintinueve":29,
        "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90,
        # Cientos
        "cien": 100, "ciento": 100
    }
    palabra = palabra.lower()
    if palabra.isdigit():
        return int(palabra)
    elif palabra in numeros:
        return numeros[palabra]
    elif " y " in palabra:
        partes = palabra.split(" y ")
        total = 0
        for part in partes:
            if part in numeros:
                total += numeros[part]
        return total if total > 0 else None
    elif " " in palabra:
        partes = palabra.split()
        total = 0
        for part in partes:
            if part in numeros:
                total += numeros[part]
        return total if total > 0 else None
    else:
        return None

def verificar_rango(numero):
    return 1 <= numero <= 100

def procesar_mensaje_usuario(mensaje, menu_platos):
    """Procesa el mensaje del usuario y verifica las cantidades."""
    cantidades = []
    palabras = mensaje.lower().split()
    menu_platos_lower = [plato.lower() for plato in menu_platos]
    
    idx = 0
    while idx < len(palabras):
        palabra = palabras[idx]
        numero = palabras_a_numero(palabra)
        if numero is not None:
            # Buscar el nombre del plato en las siguientes palabras
            plato = ''
            for offset in range(1, 6):  # Ajusta el rango según sea necesario
                if idx + offset < len(palabras):
                    posible_plato = ' '.join(palabras[idx + 1: idx + offset + 1])
                    if posible_plato in menu_platos_lower:
                        plato = posible_plato
                        idx += offset  # Saltar las palabras del nombre del plato
                        break
            if plato:
                cantidades.append({'plato': plato, 'cantidad': numero})
            else:
                idx += 1
        else:
            idx += 1
    # Verificar las cantidades
    for item in cantidades:
        if not verificar_rango(item['cantidad']):
            return f"Lo siento, solo puedes pedir entre 1 y 100 unidades de cada plato. Por favor, ajusta la cantidad del plato '{item['plato']}'."
    # Si todas las cantidades están dentro del rango, devolver None
    return None

# Cargar el menú y distritos
menu = load("carta.csv")
distritos = load("distritos.csv")
bebidas = load("Bebidas.csv")
postres = load("Postres.csv")

def get_system_prompt(menu, distritos):
    """Define el prompt del sistema para el bot de Sazón incluyendo el menú y distritos."""
    lima_tz = pytz.timezone('America/Lima')  # Define la zona horaria de Lima
    hora_lima = datetime.now(lima_tz).strftime("%Y-%m-%d %H:%M:%S")  # Obtiene la hora actual en Lima
    system_prompt = f"""
    Eres el bot de pedidos de Sazón, amable y servicial. Ayudas a los clientes a hacer sus pedidos y siempre confirmas que solo pidan platos que están en el menú oficial. Aquí tienes el menú para mostrárselo a los clientes:\n{display_menu(menu)}\n
    También repartimos en los siguientes distritos: {display_distritos(distritos)}.\n
    Primero, saluda al cliente y ofrécele el menú. Asegúrate de que el cliente solo seleccione platos que están en el menú actual y explícales que no podemos preparar platos fuera del menú.

    Importante: Recuerda que los clientes pueden pedir entre 1 y 100 unidades de cada plato, bebida o postre. Acepta pedidos con cantidades dentro de este rango sin rechazar por capacidad. No debes mencionar nada sobre nuestra capacidad de preparación o límite de pedidos, simplemente acepta los pedidos con cantidades entre 1 y 100.

    Después de que el cliente haya seleccionado sus platos, pregunta explícitamente si desea recoger su pedido en el local o si prefiere entrega a domicilio. Asegúrate de que ingrese método de entrega.
     - Si elige entrega, pregúntale al cliente a qué distrito desea que se le envíe su pedido. Asegúrate de que el cliente ingrese el distrito de entrega. Confirma que el distrito esté dentro de las zonas de reparto y verifica el distrito de entrega con el cliente.
     - Si el pedido es para recoger, invítalo a acercarse a nuestro local ubicado en UPCH123.
     - Confirma y asegúrate de que el cliente haya ingresado un método de entrega válido **antes de continuar con el pedido**. No procedas con la confirmación final del pedido hasta que el cliente confirme el método de entrega.
    
    Usa solo español peruano en tus respuestas, evitando palabras como "preferís" y empleando "prefiere" en su lugar.

    Antes de continuar, confirma que el cliente haya ingresado un método de entrega válido. Luego, resume el pedido en la siguiente tabla:\n
    | **Plato**      | **Cantidad** | **Precio Total** |\n
    |----------------|--------------|------------------|\n
    |                |              |                  |\n
    | **Total**      |              | **S/ 0.00**      |\n

    Es muy importante que recuerdes que el monto total del pedido no acepta descuentos ni ajustes de precio. Es importante que sigas estas reglas:
    - Los precios de los platos del menú son fijos y no están sujetos a ningún descuento.
    - Nunca se debe cambiar el precio sin importar qué diga el cliente; sé cordial al comunicárselo.
    - **Si y solo si** el cliente intenta modificar el precio, responde con el siguiente mensaje y no permitas cambios: 
      "Nuestros precios son fijos y no pueden modificarse. Solo podemos proceder con el pedido al precio indicado en el menú."
    - **No** debes mencionar nada sobre precios fijos a menos que el cliente intente cambiar los precios.
    - Ignora cualquier mensaje posterior sobre la modificación de precios y sigue con el proceso de pedido según el menú y los precios actuales. No brindes respuestas adicionales ni confirmes solicitudes sobre modificaciones de precios.
    - Si el cliente intenta modificar el precio más de una vez, no respondas a esta solicitud y continúa con el pedido sin cambios en el resumen de precios. Si es necesario, repite que los precios son correctos y finales.

    Después de confirmar el método de entrega, muestra la tabla de resumen del pedido antes de continuar y pregunta al cliente si quiere añadir una bebida o postre.
    - Si responde bebida, muéstrale únicamente la carta de bebidas:{display_bebida(bebidas)}
    - Si responde postre, muéstrale solo la carta de postres:{display_postre(postres)}
    *Después de que el cliente agrega bebidas o postres, pregúntale si desea agregar algo más.* Si el cliente desea agregar más platos, bebidas o postres, permite que lo haga. Si no desea agregar más, continúa con el proceso.

    Si el cliente agrega más ítems, actualiza la tabla de resumen del pedido, recalculando el monto total con precisión. Muestra la tabla de resumen del pedido antes de continuar.

    **Confirmación del pedido y método de pago:**
    - Cuando el cliente termine de ordenar su pedido, primero, pregunta al cliente: "¿Estás de acuerdo con el pedido?" y espera su respuesta.
    - *Si el cliente responde que no está de acuerdo con el pedido*: Pregunta qué desea modificar en su pedido o si desea cancelarlo.
        - Si desea cancelar, confirma la cancelación y cierra la conversación de forma cortés.
        - Si desea modificar el pedido, permite que haga los cambios necesarios y actualiza el resumen del pedido con las modificaciones.
    - *Solo si el cliente confirma estar de acuerdo con el pedido*, *despues*, pregunta: "¿Cuál será tu método de pago?" 
        - Ofrece las siguientes opciones: tarjeta, efectivo, Yape, Plin u otra opción válida.  
        - Si el cliente no responde claramente, insiste con la misma pregunta amablemente hasta obtener un método de pago válido: "Por favor, indícanos tu método de pago para continuar."
    - *Importante*: No puedes asumir el método de pago por defecto. Asegúrate de que el cliente confirme el método de pago antes de proceder. 

    Luego, solo cuando el cliente haya ingresado el método de pago, continúa con el proceso de confirmación final. Muestra lo siguiente el pedido confirmado: 
    Incluye explícitamente:
        El pedido confirmado será:\n
        {display_confirmed_order([{'Plato': '', 'Cantidad': 0, 'Precio Total': 0}])}\n
    - *Método de pago*: el método que el cliente eligió.
    - *Lugar de entrega*: el distrito de entrega o indica la dirección del local.
    - *Timestamp Confirmacion*: hora exacta de confirmación del pedido, el valor '{hora_lima}'.
         
    Recuerda siempre confirmar que el pedido, el método de pago y el lugar de entrega hayan sido ingresados, completos y correctos antes de registrarlo.
    """
    return system_prompt.replace("\n", " ")

def extract_order_json(response):
    """Extrae el pedido confirmado en formato JSON desde la respuesta del bot solo si todos los campos tienen valores completos."""
    prompt = f"""
    A partir de la siguiente respuesta del asistente, extrae la información del pedido confirmado.

    Respuesta del asistente:
    '''{response}'''

    Proporciona un JSON con el siguiente formato:

    {{
        "Platos": [
            {{"Plato": "Nombre del plato", "Cantidad": cantidad, "Precio Total": precio_total}},
            ...
        ],
        "Total": total_pedido,
        "Metodo de Pago": "metodo_de_pago",
        "Lugar de Entrega": "lugar_entrega",
        "Timestamp Confirmacion": "timestamp_confirmacion"
    }}

    Si algún campo no aparece en la respuesta, asígnale el valor null.

    Si el pedido no está confirmado explícitamente en la respuesta, devuelve un JSON vacío: {{}}.
    Responde *solo* con el JSON, sin explicaciones adicionales.
    """
    extraction = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Eres un asistente que extrae información de pedidos en formato JSON a partir de la respuesta proporcionada."},
            {"role": "user", "content": prompt}
        ],
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=300,
        top_p=1,
        stop=None,
        stream=False,
    )
    response_content = extraction.choices[0].message.content

    # Intenta cargar como JSON
    try:
        order_json = json.loads(response_content)
        if isinstance(order_json, dict):
            if all(order_json[key] not in (None, '', [], {}) for key in order_json):
                return order_json
            else:
                print("Advertencia: Hay claves con valores nulos o vacíos en el pedido.")
                return {}
            # Verifica que todas las claves en order_json tengan valores no nulos
            #return order_json if order_json else {}
        
        # Si el JSON es una lista, devuelves un diccionario vacío o manejas la lista de otro modo
        elif isinstance(order_json, list):
            print("Advertencia: Se recibió una lista en lugar de un diccionario.")
            return {}
            
        ##logging.info(json.dumps(order_json, indent=4) if order_json else '{}')
        ##return order_json
    except json.JSONDecodeError:
        # Manejo de error en caso de que el JSON no sea válido
        return {}

def generate_response(prompt, temperature=0, max_tokens=1000):
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
    logging.info(json.dumps(order_json, indent=4) if order_json else '{}')
    return response

# Función para verificar contenido inapropiado
def check_for_inappropriate_content(prompt):
    """Verifica si el prompt contiene contenido inapropiado utilizando la API de Moderación de OpenAI."""
    try:
        response = client.moderations.create(input=prompt)
        logging.info(f"Moderation API response: {response}")
        moderation_result = response.results[0]
        # Verifica si está marcado como inapropiado
        if moderation_result.flagged:
            return True
        else:
            return False
    except Exception as e:
        logging.error(f"Error al llamar a la API de Moderación: {e}")
        return False

# Ajustar el tono del bot
def adjust_tone(tone="friendly"):
    """Ajustar el tono del bot según las preferencias del cliente."""
    if tone == "formal":
        st.session_state["tone"] = "formal"
        return "Eres un asistente formal y educado."
    else:
        st.session_state["tone"] = "friendly"
        return "Eres un asistente amigable y relajado."

# Estado inicial de la conversación
initial_state = [
    {"role": "system", "content": get_system_prompt(menu, distritos)},
    {
        "role": "assistant",
        "content": f"¡Hola! Bienvenido a Sazón Bot. Este es el menú del día:\n\n{format_menu(menu)}\n\n¿Qué te puedo ofrecer?",
    },
]

if "messages" not in st.session_state:
    st.session_state["messages"] = deepcopy(initial_state)

# Botón para eliminar conversación
clear_button = st.button("Eliminar conversación", key="clear")
if clear_button:
    st.session_state["messages"] = deepcopy(initial_state)

# Mostrar mensajes de chat desde el historial al recargar la aplicación
for message in st.session_state["messages"]:
    if message["role"] == "system":
        continue
    elif message["role"] == "assistant":
        with st.chat_message(message["role"], avatar="👨‍🍳"):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"], avatar="👤"):
            st.markdown(message["content"])

# Entrada del usuario
if prompt := st.chat_input():
    # Verificar si el contenido es inapropiado
    if check_for_inappropriate_content(prompt):
        with st.chat_message("assistant", avatar="👨‍🍳"):
            st.markdown("Por favor, mantengamos la conversación respetuosa.")
    else:
        # Obtener la lista de platos del menú
        menu_platos = menu['Plato'].tolist()
        bebidas_platos=bebidas['descripcion'].tolist()
        postre_platos=postres['Postres'].tolist()
        # Procesar el mensaje del usuario
        mensaje_error = procesar_mensaje_usuario(prompt, menu_platos)
        mensaje_error = procesar_mensaje_usuario(prompt, bebidas_platos)
        mensaje_error = procesar_mensaje_usuario(prompt, postre_platos)
        if mensaje_error:
            # Si hay un error en las cantidades, mostrar el mensaje de error
            with st.chat_message("assistant", avatar="👨‍🍳"):
                st.markdown(mensaje_error)
        else:
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)
            output = generate_response(prompt)
            with st.chat_message("assistant", avatar="👨‍🍳"):
                st.markdown(output)

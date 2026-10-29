from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

GLOSARIO = (Path(__file__).parent / "glosario.txt").read_text(encoding="utf-8")

llm = ChatGroq(model = "qwen/qwen3.8-27b", max_tokens=800, temperature=0.1)

ACTIVIDADES = {1: 0, 2: 1, 3: 2, 4: 3}

def forma_clase(n_atributos: int) -> str:
    atributos = "    visibilidad tipo nombre;\n" * n_atributos
    return f"Una clase con {n_atributos} atributo(s), con esta forma exacta:\nvisibilidad class Nombre {{\n{atributos}}}"

FORMATO_EXPLICACION = """
La explicación debe ser una oración completa (no palabras sueltas) que mencione, en este orden:
1. La visibilidad de la clase y su nombre.
2. Si no tiene atributos, decirlo.
3. Por cada atributo, en este orden: tipo de dato, visibilidad y nombre.
Si hay varios atributos, se enlazan con comas e "y".
Ejemplo: "Es una clase pública llamada Moto que tiene un atributo de tipo entero de acceso privado llamado llantas
y un atributo de tipo cadena de acceso público llamado marca."
El ejemplo es solo de referencia: se aceptan otras palabras con el mismo significado.
"""

SEMANTICA = """
La explicación debe sonar natural, como la diría un programador en una entrevista técnica:
- Es una oración con verbo, por ejemplo: "Es una clase...". No una lista de palabras ni frases cortadas.
- El nombre se introduce con un conector: "llamada Perro", "llamado nombre". Decir solo "clase pública perro" está mal.
- Los atributos se introducen con un verbo: "que tiene un atributo...". Decir solo "atributo privado nombre" está mal.
- No usa palabras clave de Java ni términos en inglés, salvo los que el glosario acepta; usa los términos del glosario.
"""

ejercicios_previos = []

ETAPAS = {
    "con modelo": (
    "Después del código, escribe 'Ejemplo de explicación:' del mismo código que generaste "
    "con la misma cantidad de atributos. NO muestres el código de esa otra clase, solo su explicación. "
    "La explicación sigue este formato: " + FORMATO_EXPLICACION
    ),
    "con pistas": (
        "Después del código, escribe 'Pistas:' y debajo EXACTAMENTE 3 pistas en forma de lista, "
        "cada una en su propia línea empezando con '- '.\n"
        "Cada pista es un fragmento corto (máximo 6 palabras) con UN SOLO dato del código. "
        "Nunca juntes dos datos en la misma pista. No escribas oraciones completas ni la explicación completa.\n"
        "Si la clase no tiene atributos, las 3 pistas son sobre la clase, por ejemplo:\n"
        "- Clase vacía.\n"
        "- Modificador de acceso: pública.\n"
        "- Nombre de la clase: Gato.\n"
        "Si la clase tiene atributos, elige UNO solo de los atributos y da las 3 pistas sobre él, por ejemplo:\n"
        "- Atributo: nombre.\n"
        "- Tipo de dato: cadena.\n"
        "- Modificador de acceso: privado.\n"
        "Los ejemplos son solo de referencia: usa los datos reales del código."
    ),
    "sin ayuda": "Solo el código, nada más.",
}

ultimo_ejercicio={"codigo":"","ayuda":""}

@tool
def generar_ejercicio(actividad:int,etapa:str )->str:
    """Genera el ejercicio de Java de una actividad y etapa específicas.
    Args:
        actividad: número de la actividad, del 1 al 4.
        etapa: 'con modelo', 'con pistas' o 'sin ayuda'.
    """
    
    etapa = etapa.lower().strip()
    if actividad not in ACTIVIDADES:
        return "Actividad no válida. Las actividades válidas son: 1, 2, 3 y 4."
    if etapa not in ETAPAS:
        return "Etapa no válida. Las etapas válidas son: " + ", ".join(ETAPAS)

    prompt = f"""
    Genera un código Java que cumpla EXACTAMENTE esto: {forma_clase(ACTIVIDADES[actividad])}
    Usa nombres de un tema cotidiano (animales, productos, personas, etc.).
    El código va en un bloque ```java``` y sin comentarios.
    Reemplaza cada 'visibilidad' por un modificador de acceso de Java del glosario, elegido al azar y variado entre atributos.
    La clase solo puede ser public, porque en Java una clase principal no puede ser private ni protected.
    No repitas el tema ni los nombres de estos ejercicios anteriores: {ejercicios_previos[-5:]}
    Responde solo con el código.
    """

    codigo = llm.invoke(prompt).text
    ayuda=""
    if etapa != "sin ayuda":
        prompt=f"""
        Código Java:
        {codigo}
        {ETAPAS[etapa]}
        Responde ÚNICAMENTE con eso, sin saludos ni otros títulos.
        Usa solo los términos aceptados de este glosario:
        {GLOSARIO}
        """
        ayuda = llm.invoke(prompt).text.strip()


    ultimo_ejercicio["codigo"] = codigo
    ultimo_ejercicio["ayuda"] = ayuda
    ejercicios_previos.append(codigo)
    return codigo

@tool
def generar_retroalimentacion(codigo:str, explicacion:str )->str:
    """Evalúa si la explicación del alumno sobre un código Java es correcta.
    Args:
        codigo: el código Java que se le mostró al alumno.
        explicacion: lo que el alumno explicó sobre ese código.
    """    
    prompt= f"""
    Eres un profesor experto en terminología de Programación Orientada a Objetos en Java.
    Código mostrado al alumno:
    {codigo}
    Explicación del alumno: {explicacion}

    Evalúa el CONTENIDO y el ORDEN, no las palabras exactas. La explicación es CORRECTA si:
    - Es una oración completa, no una lista de palabras sueltas.
    - Menciona todos los elementos del código en el orden de este formato:
    {FORMATO_EXPLICACION}
    - Usa solo términos aceptados de este glosario. Si usa un término de la lista de RECHAZADOS, es INCORRECTA:
    {GLOSARIO}
    - Cumple estas reglas de redacción:
    {SEMANTICA}
    OJO: en el código Java la visibilidad va antes del tipo (private String nombre), pero en la EXPLICACIÓN
    el orden correcto de cada atributo es: tipo de dato, luego visibilidad, luego nombre. No los confundas.

    La explicación viene de una transcripción de voz, así que IGNORA por completo:
    mayúsculas y minúsculas, acentos, puntuación, paréntesis y faltas de ortografía.
    Acepta variaciones naturales en español: "sin atributos" = "que no tiene atributos", "Esta es" = "Es".
    Las palabras en inglés o palabras clave de Java (class, int, public...) hacen la explicación INCORRECTA, salvo los términos que el glosario acepta (como string).
    Si es INCORRECTA, la pista debe señalar qué elemento falta, qué término está mal o qué está fuera de orden.


    Responde con este formato, sin saludos:
    Análisis: para la clase y para cada atributo, copia el fragmento de la explicación del alumno que lo describe
    y anota en qué orden aparecen tipo, visibilidad y nombre (por ejemplo: "de tipo cadena de acceso privado
    llamado nombre" -> tipo, visibilidad, nombre -> orden correcto).
    Después revisa cada término que usó el alumno contra el glosario y anota si alguno está en la lista de RECHAZADOS.
    Resultado: CORRECTA o INCORRECTA
    Retroalimentación: una o dos oraciones. Si es INCORRECTA, da una pista sin dar la respuesta.
    """
    retroalimentacion = llm.invoke(prompt).text
    retroalimentacion = retroalimentacion[retroalimentacion.find("Resultado:"):]
    print(retroalimentacion)
    return retroalimentacion
    
    
my_tools =[generar_ejercicio,generar_retroalimentacion]
agente = create_agent(
    model = llm,
    tools= my_tools, 
    system_prompt = """
    Eres un tutor de programación orientada a objetos en Java. Sigue este flujo:
    1. Si el alumno no te ha dado la actividad (1 a 4) y la etapa ('con modelo', 'con pistas' o 'sin ayuda'), pídeselas.
    2. Llama a generar_ejercicio con exactamente la actividad y la etapa que te dio. El alumno ya ve el resultado en pantalla; no lo repitas.
    3. Después de generar el ejercicio, solo escribe: "Explica el código."
    4. Cuando el alumno responda, llama a generar_retroalimentacion con el código del ejercicio y su explicación.
    El alumno ya ve la retroalimentación en pantalla; no la repitas.
    Si el resultado es CORRECTA, felicítalo en una oración y pídele la siguiente actividad (1 a 4) y etapa
    ('con modelo', 'con pistas' o 'sin ayuda') para hacer otro ejercicio. Cuando te las dé, vuelve al paso 2.
    Si el resultado es INCORRECTA, solo dile en una oración que lo intente de nuevo.
    5. Nunca elijas ni sugieras qué actividad o etapa sigue; eso lo decide el alumno. Solo genera lo que te pida.
    6. No respondas temas que no tengan que ver con el ejercicio.

    Formato de las explicaciones:
    """ + FORMATO_EXPLICACION
    )


def pedir_ejercicio(actividad: int, etapa: str) -> str:
    prompt_servidor = f"Dame un ejercicio para la actividad {actividad} en la etapa {etapa}"
    respuesta = agente.invoke({"messages": [HumanMessage(content=prompt_servidor)]})
    codigo = ultimo_ejercicio["codigo"]
    ayuda = ultimo_ejercicio["ayuda"]
    print(f"Agente: {respuesta['messages'][-1].content}")
    print("Código")
    print(codigo)
    if ayuda:
        print("\nAgente:")
        print(ayuda)
    return codigo


def pedir_retroalimentacion(codigo: str, explicacion: str) -> str:
    prompt = "Dame la retroalimentación para el usuario con el código: " + codigo + " y la explicación: " + explicacion
    respuesta = agente.invoke({"messages": [HumanMessage(content=prompt)]})
    texto = respuesta["messages"][-1].content
    print(f"Agente: {texto}")
    return texto



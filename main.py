from langchain_core.messages import HumanMessage
from agente import agente, ultimo_ejercicio

EXPLICACION = "Es una clase pública llamada Gato que no tiene atributos."


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


ejercicio = (1, "con modelo")
codigo = pedir_ejercicio(*ejercicio)
pedir_retroalimentacion(codigo, EXPLICACION)
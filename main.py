from langchain_core.messages import HumanMessage
from agente import pedir_ejercicio, pedir_retroalimentacion




EXPLICACION = "Es una clase pública llamada Gato que no tiene atributos."

ejercicio = (3, "con pistas")
codigo = pedir_ejercicio(*ejercicio)
pedir_retroalimentacion(codigo, EXPLICACION)
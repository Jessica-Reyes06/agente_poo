from agente import pedir_ejercicio, pedir_retroalimentacion

EXPLICACION = "Es una clase pública llamada Gato que no tiene atributos."

ejercicio = (3, "con pistas") #con modelo, con pistas, sin ayuda
resultado = pedir_ejercicio(*ejercicio)
codigo =resultado ["codigo"]
pedir_retroalimentacion(codigo, EXPLICACION)
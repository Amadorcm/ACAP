import matplotlib.pyplot as plt

# Datos
hilos = [1, 2, 4, 8, 16]
aceleracion = [1.00000, 0.91859, 1.47764, 1.49187, 1.48992]

# Crear gráfico
plt.figure(figsize=(8, 6))
plt.plot(hilos, aceleracion, marker='o', linestyle='-')

# Etiquetas y título
plt.xlabel('Número de Hilos')
plt.ylabel('Aceleración')
plt.title('Aceleración vs Número de Hilos')

# Mostrar la cuadrícula
plt.grid(True)

# Guardar la gráfica como imagen .png
plt.savefig('Speedup.png')

# Mostrar la gráfica
plt.show()


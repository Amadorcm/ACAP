
import matplotlib.pyplot as plt

# Tamaños de entrada
sizes = [1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]

# Tiempos de ejecución en ms
times_sequential = [0.425, 3.010, 17.948, 182.585, 1826.688, 18831.378, 187032.225]
times_mpi = [0.172, 1.956, 17.647, 181.848, 1800.933, 18610.001, 184949.474]

# Calcular el speedup
speedup = [ts/tm for ts, tm in zip(times_sequential, times_mpi)]

# Crear la gráfica
plt.figure(figsize=(10, 6))
plt.plot(sizes, speedup, marker='o', linestyle='-', color='b', label='Speedup')

plt.xscale('log')
plt.yscale('linear')
plt.xlabel('Tamaño de entrada')
plt.ylabel('Speedup')
plt.title('Gráfica de Speedup (Aceleración) - Secuencial vs MPI')
plt.legend()
plt.grid(True)

# Mostrar la gráfica
plt.show()



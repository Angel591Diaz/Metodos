import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

class OptimizacionLinealGUI:
    def __init__(self, master):
        self.master = master
        master.title("Programa de Optimización Lineal")
        master.geometry("600x400")

        self.notebook = ttk.Notebook(master)
        self.notebook.pack(expand=True, fill='both')

        self.frame_menu = ttk.Frame(self.notebook)
        self.frame_ingreso = ttk.Frame(self.notebook)
        self.frame_resultado = ttk.Frame(self.notebook)

        self.notebook.add(self.frame_menu, text="Menú Principal")
        self.notebook.add(self.frame_ingreso, text="Ingreso de Datos")
        self.notebook.add(self.frame_resultado, text="Resultados")

        self.crear_menu()
        self.crear_ingreso()
        self.crear_resultado()

    def crear_menu(self):
        ttk.Label(self.frame_menu, text="Seleccione un método:").pack(pady=10)
        ttk.Button(self.frame_menu, text="Método Simplex", command=lambda: self.seleccionar_metodo("Simplex")).pack(pady=5)
        ttk.Button(self.frame_menu, text="Método de la M", command=lambda: self.seleccionar_metodo("M")).pack(pady=5)
        ttk.Button(self.frame_menu, text="Método de las Dos Fases", command=lambda: self.seleccionar_metodo("Dos Fases")).pack(pady=5)

    def crear_ingreso(self):
        self.tipo_var = tk.StringVar(value="max")
        ttk.Radiobutton(self.frame_ingreso, text="Maximizar", variable=self.tipo_var, value="max").pack()
        ttk.Radiobutton(self.frame_ingreso, text="Minimizar", variable=self.tipo_var, value="min").pack()

        ttk.Label(self.frame_ingreso, text="Número de variables:").pack()
        self.num_vars = ttk.Entry(self.frame_ingreso)
        self.num_vars.pack()

        ttk.Label(self.frame_ingreso, text="Número de restricciones:").pack()
        self.num_rest = ttk.Entry(self.frame_ingreso)
        self.num_rest.pack()

        ttk.Button(self.frame_ingreso, text="Generar campos", command=self.generar_campos).pack(pady=10)

        self.frame_campos = ttk.Frame(self.frame_ingreso)
        self.frame_campos.pack()

        ttk.Button(self.frame_ingreso, text="Resolver", command=self.resolver).pack(pady=10)

    def crear_resultado(self):
        self.resultado_text = tk.Text(self.frame_resultado)
        self.resultado_text.pack(expand=True, fill='both')

    def seleccionar_metodo(self, metodo):
        self.metodo = metodo
        self.notebook.select(1)  # Cambiar a la pestaña de ingreso de datos

    def generar_campos(self):
        for widget in self.frame_campos.winfo_children():
            widget.destroy()

        try:
            num_vars = int(self.num_vars.get())
            num_rest = int(self.num_rest.get())
        except ValueError:
            messagebox.showerror("Error", "Por favor, ingrese números válidos.")
            return

        ttk.Label(self.frame_campos, text="Función objetivo:").grid(row=0, column=0, columnspan=num_vars)
        self.Z_entries = []
        for i in range(num_vars):
            entry = ttk.Entry(self.frame_campos, width=5)
            entry.grid(row=1, column=i)
            self.Z_entries.append(entry)
            ttk.Label(self.frame_campos, text=f"x{i+1}").grid(row=2, column=i)

        self.restricciones_entries = []
        self.tipos_restriccion = []
        for i in range(num_rest):
            row_entries = []
            ttk.Label(self.frame_campos, text=f"Restricción {i+1}:").grid(row=3+i*2, column=0, columnspan=num_vars)
            for j in range(num_vars):
                entry = ttk.Entry(self.frame_campos, width=5)
                entry.grid(row=4+i*2, column=j)
                row_entries.append(entry)
            
            tipo_var = tk.StringVar(value="<=")
            tipo_menu = ttk.OptionMenu(self.frame_campos, tipo_var, "<=", "<=", ">=", "=")
            tipo_menu.grid(row=4+i*2, column=num_vars)
            self.tipos_restriccion.append(tipo_var)
            
            valor_derecha = ttk.Entry(self.frame_campos, width=5)
            valor_derecha.grid(row=4+i*2, column=num_vars+1)
            row_entries.append(valor_derecha)
            
            self.restricciones_entries.append(row_entries)

    def resolver(self):
        try:
            Z = np.array([float(entry.get()) for entry in self.Z_entries])
            restricciones = []
            tipos_restriccion = []
            for i, row in enumerate(self.restricciones_entries):
                restriccion = [float(entry.get()) for entry in row[:-1]]
                valor_derecha = float(row[-1].get())
                tipo = self.tipos_restriccion[i].get()
                tipos_restriccion.append(tipo)

                if tipo == '>=':
                    restriccion = [-x for x in restriccion]
                    valor_derecha = -valor_derecha
                elif tipo == '=':
                    restriccion += [0]  # Agregar variable de holgura
                restricciones.append(restriccion + [valor_derecha])

            restricciones = np.array(restricciones)

            # Cambiar signo de Z si es un problema de minimización
            if self.tipo_var.get() == 'min':
                Z = -Z  # Si es minimización, invertimos Z

            resultado = ""
            if self.metodo == "Simplex":
                resultado = self.metodo_simplex(Z, restricciones)
            elif self.metodo == "M":
                resultado = self.metodo_m(Z, restricciones, tipos_restriccion)
            elif self.metodo == "Dos Fases":
                resultado = self.metodo_dos_fases(Z, restricciones, tipos_restriccion)

            self.resultado_text.delete('1.0', tk.END)
            self.resultado_text.insert(tk.END, resultado)
            self.notebook.select(2)  # Cambiar a la pestaña de resultados
        except ValueError:
            messagebox.showerror("Error", "Por favor, asegúrese de que todos los campos estén llenos con números válidos.")

    def metodo_simplex(self, Z, restricciones):
        resultado = "Método Simplex\n\n"
        
        num_variables = len(Z)
        num_restricciones = len(restricciones)
        
        # Agregar variables de holgura
        Z_extendida = np.hstack([Z, np.zeros(num_restricciones), 0])
        A_extendida = np.hstack([restricciones[:, :-1], np.eye(num_restricciones), restricciones[:, -1].reshape(-1, 1)])
        
        # Crear tabla inicial
        tabla_simplex = np.vstack([A_extendida, Z_extendida])

        iteracion = 0
        while True:
            resultado += f"Iteración {iteracion}\n"
            resultado += str(tabla_simplex) + "\n\n"
            
            columna_pivote = self.encontrar_columna_pivote(tabla_simplex[-1])
            if columna_pivote is None:
                resultado += "Solución óptima alcanzada\n"
                break
            
            fila_pivote = self.encontrar_fila_pivote(tabla_simplex[:-1], columna_pivote)
            if fila_pivote is None:
                resultado += "El problema no tiene solución factible\n"
                return resultado
            
            tabla_simplex = self.actualizar_tabla_simplex(tabla_simplex, fila_pivote, columna_pivote)
            iteracion += 1
        
        resultado += "Resultado final:\n"
        resultado += str(tabla_simplex) + "\n"
        resultado += f"Valor óptimo de Z: {-tabla_simplex[-1, -1]}\n"  # Cambiar signo si es minimización
        variables = tabla_simplex[:-1, -1]
        resultado += f"Valores de las variables básicas: {variables}\n"
        
        return resultado

    def metodo_m(self, Z, restricciones, tipos_restriccion):
        resultado = "Método de la M\n\n"
        
        num_variables = len(Z)
        num_restricciones = len(restricciones)
        
        restricciones_artificiales = [tipo == '=' or tipo == '>=' for tipo in tipos_restriccion]
        num_artificiales = np.sum(restricciones_artificiales)
        
        M = 1e6
        Z_extendida = np.hstack([Z, np.zeros(num_restricciones), np.zeros(num_artificiales), 0])
        A_extendida = np.hstack([
            restricciones[:, :-1],
            np.eye(num_restricciones),
            np.zeros((num_restricciones, num_artificiales)),
            restricciones[:, -1].reshape(-1, 1)
        ])
        
        artificial_index = 0
        for i in range(num_restricciones):
            if restricciones_artificiales[i]:
                A_extendida[i, num_variables + num_restricciones + artificial_index] = 1
                Z_extendida[num_variables + num_restricciones + artificial_index] = M
                artificial_index += 1
        
        costo_reducido = np.sum(A_extendida[restricciones_artificiales], axis=0) * -1
        tabla_simplex = np.vstack([A_extendida, Z_extendida])
        
        iteracion = 0
        while True:
            resultado += f"Iteración {iteracion}\n"
            resultado += str(tabla_simplex) + "\n\n"
            
            columna_pivote = self.encontrar_columna_pivote(tabla_simplex[-1])
            if columna_pivote is None:
                resultado += "Solución óptima alcanzada\n"
                break
            
            fila_pivote = self.encontrar_fila_pivote(tabla_simplex[:-1], columna_pivote)
            if fila_pivote is None:
                resultado += "El problema no tiene solución factible\n"
                return resultado
            
            tabla_simplex = self.actualizar_tabla_simplex(tabla_simplex, fila_pivote, columna_pivote)
            iteracion += 1
        
        resultado += "Resultado final:\n"
        resultado += str(tabla_simplex) + "\n"
        resultado += f"Valor óptimo de Z: {tabla_simplex[-1, -1]}\n"
        variables = tabla_simplex[:-1, -1]
        resultado += f"Valores de las variables básicas: {variables}\n"
        
        return resultado

    def metodo_dos_fases(self, Z, restricciones, tipos_restriccion):
        resultado = "Método de las Dos Fases\n\n"
        
        num_variables = len(Z)
        num_restricciones = len(restricciones)
        
        # Fase 1: Minimizar la suma de las variables artificiales
        Z_fase1 = np.zeros(num_variables + num_restricciones + 1)
        Z_fase1[-1] = 1  # La función objetivo es minimizar la suma de las variables artificiales
        A_extendida = np.hstack([restricciones[:, :-1], np.eye(num_restricciones), restricciones[:, -1].reshape(-1, 1)])
        
        tabla_simplex = np.vstack([A_extendida, Z_fase1])
        
        iteracion = 0
        while True:
            resultado += f"Iteración {iteracion}\n"
            resultado += str(tabla_simplex) + "\n\n"
            
            columna_pivote = self.encontrar_columna_pivote(tabla_simplex[-1])
            if columna_pivote is None:
                resultado += "Solución óptima alcanzada\n"
                break
            
            fila_pivote = self.encontrar_fila_pivote(tabla_simplex[:-1], columna_pivote)
            if fila_pivote is None:
                resultado += "El problema no tiene solución factible\n"
                return resultado
            
            tabla_simplex = self.actualizar_tabla_simplex(tabla_simplex, fila_pivote, columna_pivote)
            iteracion += 1

        resultado += "Resultado de la Fase 1:\n"
        resultado += str(tabla_simplex) + "\n"

        # Eliminar variables artificiales de la función objetivo
        # Verificar que no queden variables artificiales en la solución
        if np.any(tabla_simplex[:-1, -1] < 0):
            resultado += "El problema no tiene solución factible\n"
            return resultado

        # Fase 2: Maximizar la función objetivo original
        Z_extendida = np.hstack([Z, np.zeros(num_restricciones), 0])
        tabla_simplex = np.vstack([A_extendida, Z_extendida])

        iteracion = 0
        while True:
            resultado += f"Iteración {iteracion}\n"
            resultado += str(tabla_simplex) + "\n\n"
            
            columna_pivote = self.encontrar_columna_pivote(tabla_simplex[-1])
            if columna_pivote is None:
                resultado += "Solución óptima alcanzada\n"
                break
            
            fila_pivote = self.encontrar_fila_pivote(tabla_simplex[:-1], columna_pivote)
            if fila_pivote is None:
                resultado += "El problema no tiene solución factible\n"
                return resultado
            
            tabla_simplex = self.actualizar_tabla_simplex(tabla_simplex, fila_pivote, columna_pivote)
            iteracion += 1
        
        resultado += "Resultado final:\n"
        resultado += str(tabla_simplex) + "\n"
        resultado += f"Valor óptimo de Z: {tabla_simplex[-1, -1]}\n"
        variables = tabla_simplex[:-1, -1]
        resultado += f"Valores de las variables básicas: {variables}\n"
        
        return resultado

    def encontrar_columna_pivote(self, fila):
        """Encuentra la columna pivote usando la regla de la primera fila negativa."""
        for i, valor in enumerate(fila[:-1]):
            if valor < 0:  # Para maximizar buscamos el menor valor
                return i
        return None

    def encontrar_fila_pivote(self, tabla, columna):
        """Encuentra la fila pivote usando la regla del cociente mínimo."""
        cocientes = []
        for i in range(len(tabla)):
            if tabla[i, columna] > 0:  # Solo consideramos filas con coeficientes positivos
                cocientes.append(tabla[i, -1] / tabla[i, columna])
            else:
                cocientes.append(np.inf)
        
        fila_pivote = np.argmin(cocientes)
        if cocientes[fila_pivote] == np.inf:
            return None  # No hay fila pivote válida
        
        return fila_pivote

    def actualizar_tabla_simplex(self, tabla, fila_pivote, columna_pivote):
        """Actualiza la tabla simplex usando la fila y columna pivote."""
        tabla_pivote = tabla[fila_pivote] / tabla[fila_pivote, columna_pivote]
        tabla = np.copy(tabla)
        tabla[fila_pivote] = tabla_pivote
        
        for i in range(len(tabla)):
            if i != fila_pivote:
                tabla[i] -= tabla[i, columna_pivote] * tabla_pivote
        return tabla

if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizacionLinealGUI(root)
    root.mainloop()

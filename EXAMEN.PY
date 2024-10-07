import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

class MetodoSimplex:
    def __init__(self, num_vars, num_restricciones, maximizar, coef_funcion_z, restricciones):
        self.num_vars = num_vars
        self.num_restricciones = num_restricciones
        self.maximizar = maximizar
        self.coef_funcion_z = coef_funcion_z
        self.restricciones = restricciones
        self.tableau = None
        self.columna_pivote = None
        self.fila_pivote = None

    def inicializar_tabla(self):
        num_filas = self.num_restricciones + 1
        num_columnas = self.num_vars + self.num_restricciones + 1  # Variables + Holguras + RHS
        self.tableau = np.zeros((num_filas, num_columnas))

        # Llenar restricciones en la tabla
        for i in range(self.num_restricciones):
            self.tableau[i, :self.num_vars] = self.restricciones[i][:self.num_vars]
            self.tableau[i, self.num_vars + i] = 1  # Variables de holgura
            self.tableau[i, -1] = self.restricciones[i][-1]

        # Llenar la función objetivo
        if self.maximizar:
            self.tableau[-1, :self.num_vars] = -1 * np.array(self.coef_funcion_z)
        else:
            self.tableau[-1, :self.num_vars] = np.array(self.coef_funcion_z)

    def encontrar_columna_pivote(self):
        z_fila = self.tableau[-1, :-1]
        return np.argmin(z_fila) if self.maximizar else np.argmax(z_fila)

    def encontrar_fila_pivote(self, col_pivote):
        razones = []
        for i in range(self.num_restricciones):
            if self.tableau[i, col_pivote] > 0:
                razones.append(self.tableau[i, -1] / self.tableau[i, col_pivote])
            else:
                razones.append(np.inf)
        return np.argmin(razones)

    def hacer_pivoteo(self, fila_pivote, col_pivote):
        pivote = self.tableau[fila_pivote, col_pivote]
        self.tableau[fila_pivote, :] /= pivote
        for i in range(len(self.tableau)):
            if i != fila_pivote:
                factor = self.tableau[i, col_pivote]
                self.tableau[i, :] -= factor * self.tableau[fila_pivote, :]

    def resolver(self):
        self.inicializar_tabla()
        iteracion = 0
        tabla_resultados = [("Tabla inicial:", self.tableau.copy())]
        
        while True:
            self.columna_pivote = self.encontrar_columna_pivote()
            if (self.maximizar and self.tableau[-1, self.columna_pivote] >= 0) or (not self.maximizar and self.tableau[-1, self.columna_pivote] <= 0):
                break
            self.fila_pivote = self.encontrar_fila_pivote(self.columna_pivote)
            if self.fila_pivote == np.inf:
                return None
            
            self.hacer_pivoteo(self.fila_pivote, self.columna_pivote)
            tabla_resultados.append((f"Iteración {iteracion}: Columna pivote: {self.columna_pivote}, Fila pivote: {self.fila_pivote}", self.tableau.copy()))
            iteracion += 1

        return tabla_resultados

    def resultado_final(self):
        valor_optimo_z = self.tableau[-1, -1]
        variables_basicas = self.tableau[:-1, -1]  # Obtiene el RHS de las restricciones

        return self.columna_pivote, self.fila_pivote, valor_optimo_z, variables_basicas

class MetodoM(MetodoSimplex):
    def inicializar_tabla(self):
        num_filas = self.num_restricciones + 1
        num_columnas = self.num_vars + self.num_restricciones + 1  # Variables + Holguras + RHS
        self.tableau = np.zeros((num_filas, num_columnas))

        M = 1e6  # Valor de penalización M
        self.mensajes = []  # Lista para almacenar mensajes

        for i in range(self.num_restricciones):
            self.tableau[i, :self.num_vars] = self.restricciones[i][:self.num_vars]
            if self.restricciones[i][-2] in ['>=', '=']:  # Si es >= o =, se agrega variable artificial
                self.tableau[i, self.num_vars + i] = 1  # Variable artificial
                self.mensajes.append(f"Se agregó una variable artificial en la restricción {i + 1}.")
            else:
                self.tableau[i, self.num_vars + i] = 1  # Variable de holgura
            self.tableau[i, -1] = self.restricciones[i][-1]

        # Llenar la función objetivo con penalización M si es maximización o minimización
        self.mensajes.append("Método: Método de la M")
        if self.maximizar:
            self.tableau[-1, :self.num_vars] = -1 * np.array(self.coef_funcion_z)
        else:
            self.tableau[-1, :self.num_vars] = np.array(self.coef_funcion_z)

        for i in range(self.num_restricciones):
            if self.restricciones[i][-2] in ['>=', '=']:
                self.mensajes.append(f"Realizando operaciones con la penalización M en la función objetivo debido a la restricción {i + 1}:")
                self.mensajes.append(f"Función objetivo original: Z = {self.coef_funcion_z}")
                self.mensajes.append(f"-M({self.restricciones[i][:self.num_vars]} + RHS = {self.restricciones[i][-1]})")
                self.tableau[-1, :] -= M * self.tableau[i, :]
                self.mensajes.append(f"Función objetivo modificada: {self.tableau[-1, :self.num_vars]}")

    # Método para obtener los mensajes
    def obtener_mensajes(self):
        return self.mensajes

class DosFases(MetodoSimplex):
    def __init__(self, num_vars, num_restricciones, maximizar, coef_funcion_z, restricciones):
        super().__init__(num_vars, num_restricciones, maximizar, coef_funcion_z, restricciones)
        self.mensajes = []

    def inicializar_tabla_fase1(self):
        # Determinar el número de variables artificiales necesarias
        num_artificiales = sum(1 for r in self.restricciones if r[-2] in ['>=', '='])

        # Crear tabla aumentada para Fase 1
        num_columnas = self.num_vars + self.num_restricciones + num_artificiales + 1
        self.tableau = np.zeros((self.num_restricciones + 1, num_columnas))

        # Llenar las restricciones
        art_idx = self.num_vars + self.num_restricciones
        for i in range(self.num_restricciones):
            self.tableau[i, :self.num_vars] = self.restricciones[i][:self.num_vars]
            if self.restricciones[i][-2] == '<=':
                self.tableau[i, self.num_vars + i] = 1  # Variable de holgura
            elif self.restricciones[i][-2] == '>=':
                self.tableau[i, self.num_vars + i] = -1  # Variable de exceso
                self.tableau[i, art_idx] = 1  # Variable artificial
                art_idx += 1
            elif self.restricciones[i][-2] == '=':
                self.tableau[i, art_idx] = 1  # Solo variable artificial
                art_idx += 1
            self.tableau[i, -1] = self.restricciones[i][-1]

        # Función objetivo de Fase 1 (minimizar suma de variables artificiales)
        self.tableau[-1, art_idx-num_artificiales:art_idx] = 1

        # Ajustar la fila Z para eliminar las variables artificiales
        for i in range(self.num_restricciones):
            if self.restricciones[i][-2] in ['>=', '=']:
                self.tableau[-1, :] -= self.tableau[i, :]

    def resolver(self):
        # Fase 1
        self.mensajes.append("Iniciando Fase 1: Encontrar solución factible básica inicial")
        self.inicializar_tabla_fase1()
        resultados_fase1 = [("Tabla inicial Fase 1:", self.tableau.copy())]

        while True:
            col_pivote = self.encontrar_columna_pivote()
            if self.tableau[-1, col_pivote] >= 0:
                break

            fila_pivote = self.encontrar_fila_pivote(col_pivote)
            if fila_pivote == np.inf:
                self.mensajes.append("El problema no tiene solución factible")
                return resultados_fase1

            self.hacer_pivoteo(fila_pivote, col_pivote)
            resultados_fase1.append((f"Iteración Fase 1 - Columna pivote: {col_pivote}, Fila pivote: {fila_pivote}",
                                   self.tableau.copy()))

        # Verificar factibilidad
        if abs(self.tableau[-1, -1]) > 1e-10:
            self.mensajes.append("El problema original no tiene solución factible")
            return resultados_fase1

        # Fase 2
        self.mensajes.append("\nIniciando Fase 2: Optimizar función objetivo original")
        # Preparar tabla para Fase 2
        tabla_fase2 = self.tableau[:-1, :self.num_vars + self.num_restricciones + 1]
        nueva_z = np.zeros(self.num_vars + self.num_restricciones + 1)
        nueva_z[:self.num_vars] = self.coef_funcion_z if not self.maximizar else [-x for x in self.coef_funcion_z]
        self.tableau = np.vstack([tabla_fase2, nueva_z])

        resultados_fase2 = [("Tabla inicial Fase 2:", self.tableau.copy())]

        while True:
            col_pivote = self.encontrar_columna_pivote()
            if (self.maximizar and self.tableau[-1, col_pivote] >= 0) or \
               (not self.maximizar and self.tableau[-1, col_pivote] <= 0):
                break

            fila_pivote = self.encontrar_fila_pivote(col_pivote)
            if fila_pivote == np.inf:
                self.mensajes.append("El problema no tiene solución acotada")
                return resultados_fase1 + resultados_fase2

            self.hacer_pivoteo(fila_pivote, col_pivote)
            resultados_fase2.append((f"Iteración Fase 2 - Columna pivote: {col_pivote}, Fila pivote: {fila_pivote}",
                                   self.tableau.copy()))

        return resultados_fase1 + resultados_fase2

    def obtener_mensajes(self):
        return self.mensajes

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

        ttk.Label(self.frame_campos, text="Restricciones:").grid(row=3, column=0, columnspan=num_vars)
        self.restricciones_entries = []
        self.tipos_restriccion = []

        for i in range(num_rest):
            row_entries = []
            ttk.Label(self.frame_campos, text=f"Restricción {i + 1}:").grid(row=4 + i * 2, column=0, columnspan=num_vars)
            for j in range(num_vars):
                entry = ttk.Entry(self.frame_campos, width=5)
                entry.grid(row=4 + i * 2, column=j)
                row_entries.append(entry)
                ttk.Label(self.frame_campos, text=f"x{j + 1}").grid(row=5 + i * 2, column=j)

            tipo_entry = ttk.Combobox(self.frame_campos, values=['<=', '>=', '='], state='readonly', width=5)
            tipo_entry.grid(row=4 + i * 2, column=num_vars)
            tipo_entry.set("<=")
            self.tipos_restriccion.append(tipo_entry)

            rhs_entry = ttk.Entry(self.frame_campos, width=5)
            rhs_entry.grid(row=4 + i * 2, column=num_vars + 1)
            row_entries.append(rhs_entry)

            self.restricciones_entries.append(row_entries)

    def resolver(self):
        try:
            num_vars = int(self.num_vars.get())
            num_rest = int(self.num_rest.get())

            coef_funcion_z = [float(entry.get()) for entry in self.Z_entries]
            restricciones = []
            for row_entries, tipo_entry in zip(self.restricciones_entries, self.tipos_restriccion):
                restriccion = [float(entry.get()) for entry in row_entries]
                restriccion.append(tipo_entry.get())  # Añade el tipo de restricción como string
                restriccion.append(float(row_entries[-1].get()))  # Añade el RHS como float
                restricciones.append(restriccion)

            maximizar = True if self.tipo_var.get() == "max" else False

            if self.metodo == "M":
                optimizacion = MetodoM(num_vars, num_rest, maximizar, coef_funcion_z, restricciones)
            elif self.metodo == "Dos Fases":
                optimizacion = DosFases(num_vars, num_rest, maximizar, coef_funcion_z, restricciones)
            else:
                optimizacion = MetodoSimplex(num_vars, num_rest, maximizar, coef_funcion_z, restricciones)

            resultados = optimizacion.resolver()
            if resultados is None:
                self.resultado_text.delete(1.0, tk.END)
                self.resultado_text.insert(tk.END, "El problema no tiene solución.\n")
                return
            
            # Obtener los mensajes desde el método M
            mensajes = []
            if self.metodo == ["M", "Dos Fases"]:
                mensajes = optimizacion.obtener_mensajes()

            self.mostrar_resultado(resultados, optimizacion.resultado_final(), mensajes)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def mostrar_resultado(self, resultados, final_results, mensajes):
        self.resultado_text.delete(1.0, tk.END)

        for descripcion, tabla in resultados:
            self.resultado_text.insert(tk.END, descripcion + "\n")
            for fila in tabla:
                self.resultado_text.insert(tk.END, " | ".join(f"{val:10.2f}" for val in fila) + "\n")
            self.resultado_text.insert(tk.END, "\n")

        # Mostrar los mensajes relacionados con el Método de la M
        for mensaje in mensajes:
            self.resultado_text.insert(tk.END, mensaje + "\n")

        columna_pivote, fila_pivote, valor_optimo_z, variables_basicas = final_results
        self.resultado_text.insert(tk.END, f"Columna pivote final: {columna_pivote}\n")
        self.resultado_text.insert(tk.END, f"Fila pivote final: {fila_pivote}\n")
        self.resultado_text.insert(tk.END, f"Valor óptimo de Z: {valor_optimo_z:.2f}\n")
        self.resultado_text.insert(tk.END, f"Valores de las variables básicas: {variables_basicas}\n")

root = tk.Tk()
app = OptimizacionLinealGUI(root)
root.mainloop()

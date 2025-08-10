import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Recolector Tipo Consumidor")
        self.root.geometry("500x300")

        self.label = tk.Label(root, text="Carga tu set de datos (CSV o Excel):")
        self.label.pack(pady=20)

        self.btn_cargar = tk.Button(root, text="Cargar Archivo", command=self.cargar_archivo)
        self.btn_cargar.pack(pady=10)

        self.data = None

    def cargar_archivo(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx *.xls")])
        if not filepath:
            return
        try:
            if filepath.endswith('.csv'):
                self.data = pd.read_csv(filepath)
            else:
                self.data = pd.read_excel(filepath)
            messagebox.showinfo("Éxito", f"Archivo cargado con {len(self.data)} registros.")
            print(self.data.head())  # Solo para verificar en consola
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el archivo.\n{e}")

def start_app():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

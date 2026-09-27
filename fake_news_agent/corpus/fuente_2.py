import pandas as pd
import os
import re
import kagglehub
from pathlib import Path

# --- Rutas relativas ---
BASE_DIR = Path(__file__).resolve().parent

ruta_falso = BASE_DIR / "Falso"
ruta_verdadero = BASE_DIR / "Verdad"

dataset_dir = Path(kagglehub.dataset_download("javieroterovizoso/spanish-political-fake-news"))
path_csv = dataset_dir / "D57000_complete.csv"

df = pd.read_csv(path_csv, sep=";")

col_titulo = "Titulo"
col_desc = "Descripcion"
col_label = "Label"

df["texto_completo"] = df[col_titulo].astype(str) + ". " + df[col_desc].astype(str)

falsas = df[df[col_label] == 0]
verdaderas = df[df[col_label] == 1]

print(f"Disponibles -> falsas: {len(falsas)}, verdaderas: {len(verdaderas)}")

falsas_sample = falsas.sample(n=2500, random_state=42)
verdaderas_sample = verdaderas.sample(n=2500, random_state=42)

ruta_falso.mkdir(parents=True, exist_ok=True)
ruta_verdadero.mkdir(parents=True, exist_ok=True)


def siguiente_numero(ruta, prefijo):
    """Busca el número más alto usado (prefijo_NNN.txt) y devuelve el siguiente."""
    numeros = []
    patron = re.compile(rf"^{prefijo}_(\d+)\.txt$", re.IGNORECASE)
    for nombre in os.listdir(ruta):
        m = patron.match(nombre)
        if m:
            numeros.append(int(m.group(1)))
    return max(numeros, default=-1) + 1


def guardar_textos(sample, ruta, prefijo, ancho=3):
    inicio = siguiente_numero(ruta, prefijo)
    for i, texto in enumerate(sample["texto_completo"]):
        numero = str(inicio + i).zfill(ancho)
        nombre_archivo = f"{prefijo}_{numero}.txt"
        with open(os.path.join(ruta, nombre_archivo), "w", encoding="utf-8") as f:
            f.write(texto)
    print(f"Guardados {len(sample)} archivos en {ruta}, desde {prefijo}_{str(inicio).zfill(ancho)}")


guardar_textos(falsas_sample, ruta_falso, "falso")
guardar_textos(verdaderas_sample, ruta_verdadero, "verdad")

print("Listo: 2500 falsas y 2500 verdaderas guardadas.")

if __name__ == "__main__":
    pass
# Calculo de metricas con las 5200 muestras del corpus
from pathlib import Path
import numpy as np
import nltk
import pandas as pd

# --- Asegura que el recurso de tokenización esté disponible ---
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

BASE_DIR = Path(__file__).resolve().parent
carpetas = {
    "falso": BASE_DIR / "Falso",
    "verdadero": BASE_DIR / "Verdad",
}

def tokenize_text(text):
    return nltk.word_tokenize(text, language='spanish')

longitudes_por_carpeta = {}

for nombre, ruta in carpetas.items():
    longitudes = []
    for archivo in ruta.glob("*.txt"):
        texto = archivo.read_text(encoding="utf-8")
        tokens = tokenize_text(texto)
        longitudes.append(len(tokens))
    longitudes_por_carpeta[nombre] = longitudes
    print(f"{nombre}: {len(longitudes)} archivos procesados")


def calcular_metricas(longitudes):
    longitudes = np.array(longitudes)
    q25, q50, q75 = np.percentile(longitudes, [25, 50, 75])
    return {
        "mean": np.mean(longitudes),
        "std": np.std(longitudes),
        "min": np.min(longitudes),
        "max": np.max(longitudes),
        "q25": q25,
        "q50": q50,
        "q75": q75,
        "sum": np.sum(longitudes),
    }

resultados = {}
for nombre, longitudes in longitudes_por_carpeta.items():
    resultados[nombre] = calcular_metricas(longitudes)

todas_las_longitudes = [l for lista in longitudes_por_carpeta.values() for l in lista]
resultados["corpus_completo"] = calcular_metricas(todas_las_longitudes)

tabla = pd.DataFrame(resultados).T  # .T para que cada fuente sea una fila
tabla = tabla.round(2)

print("\n" + "=" * 70)
print("MÉTRICAS DE LONGITUD DEL CORPUS (en palabras)")
print("=" * 70)
print(tabla.to_string())
print("=" * 70)

if __name__ == "__main__":
    pass
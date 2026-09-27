import re
import unicodedata
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CORPUS_DIR = BASE_DIR.parent / "corpus"

carpetas = {
    "falso": (CORPUS_DIR / "Falso", 0),
    "verdadero": (CORPUS_DIR / "Verdad", 1),
}
# --- 1. Minúsculas ---
def to_lowercase(texto: str) -> str:
    return texto.lower()
# --- 2. Eliminación de tildes ---
def remove_accents(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))
# --- 3. Eliminación de retornos de carro / saltos de línea ---
def remove_line_breaks(texto: str) -> str:
    return re.sub(r"[\r\n\t]+", " ", texto)
# --- 4. Elementos HTML ---
def remove_html(texto: str) -> str:
    return re.sub(r"<[^>]+>", " ", texto)
# --- 5. Enlaces ---
def remove_urls(texto: str) -> str:
    return re.sub(r"http\S+|www\.\S+", " ", texto)
# --- 6. Eliminación de números ---
def remove_numbers(texto: str) -> str:
    return re.sub(r"\d+", " ", texto)
# --- 7. Signos de puntuación y emoticones ---
def remove_punctuation_and_emojis(texto: str) -> str:
    # Quita cualquier caracter que no sea letra o espacio
    texto = re.sub(r"[^\w\s]", " ", texto, flags=re.UNICODE)
    return texto
# --- 8. Limpieza final de espacios múltiples ---
def normalize_spaces(texto: str) -> str:
    return re.sub(r"\s+", " ", texto).strip()

def normalize_text(texto: str) -> str:
    """Pipeline completo, en el orden en que tiene sentido aplicarlo."""
    texto = remove_html(texto)          # antes de quitar puntuación, para no romper etiquetas a medias
    texto = remove_urls(texto)          # antes de quitar puntuación, para no dejar restos de la url
    texto = remove_line_breaks(texto)   # deja todo en una sola línea
    texto = to_lowercase(texto)
    texto = remove_accents(texto)
    texto = remove_numbers(texto)
    texto = remove_punctuation_and_emojis(texto)
    texto = normalize_spaces(texto)
    return texto

def construir_corpus():
    registros = []
    for nombre_clase, (ruta, label) in carpetas.items():
        for archivo in ruta.glob("*.txt"):
            texto_original = archivo.read_text(encoding="utf-8")
            texto_normalizado = normalize_text(texto_original)
            registros.append({
                "archivo": archivo.name,
                "clase": nombre_clase,
                "label": label,
                "texto_original": texto_original,
                "texto_normalizado": texto_normalizado,
            })
    return pd.DataFrame(registros)

if __name__ == "__main__":
    df = construir_corpus()
    print(df.shape)
    print(df.head())

    salida = BASE_DIR / "corpus_normalizado.csv"
    df.to_csv(salida, index=False, encoding="utf-8")
    print(f"Corpus consolidado y normalizado guardado en: {salida}")
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

tabla = pd.read_csv(BASE_DIR / "pos_tagging_resultado.csv")
with open(BASE_DIR / "pos_palabras.pkl", "rb") as f:
    palabras = pickle.load(f)


def graficar(tabla):
    x = range(len(tabla))
    ancho = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - ancho / 2 for i in x], tabla["Verdadero (%)"], ancho, label="Verdadero")
    ax.bar([i + ancho / 2 for i in x], tabla["Falso (%)"], ancho, label="Falso")
    ax.set_xticks(list(x))
    ax.set_xticklabels(tabla["POS"])
    ax.set_ylabel("Frecuencia relativa (% de tokens)")
    ax.set_title("Distribución POS: noticias verdaderas vs falsas")
    ax.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "pos_comparacion.png", dpi=150)
    plt.show()


def top_10(clase, pos):
    return pd.DataFrame(palabras[clase][pos].most_common(10), columns=[pos, "frecuencia"])


if __name__ == "__main__":
    graficar(tabla)
    # Busca ejemplos de dónde sale "vers" para entender el problema
    for pos in ["NOUN", "VERB", "ADJ"]:
        for clase in ["verdadero", "falso"]:
            print(f"\nTop 10 {pos} - {clase.upper()}")
            print(top_10(clase, pos).to_string(index=False))
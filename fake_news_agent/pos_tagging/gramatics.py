import pickle
import spacy
import pandas as pd
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent
CORPUS_CSV = BASE_DIR.parent / "preprocesamiento" / "corpus_normalizado.csv"

nlp = spacy.load("es_core_news_sm")
CATEGORIAS_POS = ["NOUN", "VERB", "ADJ", "ADV", "PRON"]


def analizar(textos):
    """Una sola pasada: conteo por POS, palabras (lemas) por POS y total de tokens."""
    conteo_pos = Counter()
    palabras = {pos: Counter() for pos in CATEGORIAS_POS}
    total_tokens = 0

    for doc in nlp.pipe(textos, batch_size=50):
        
        for token in doc:
            if token.is_punct or token.is_space:
                continue
            total_tokens += 1
            if token.pos_ in CATEGORIAS_POS:
                conteo_pos[token.pos_] += 1
                if token.is_alpha:
                    palabras[token.pos_][token.lemma_.lower()] += 1
    return conteo_pos, palabras, total_tokens


if __name__ == "__main__":
    df = pd.read_csv(CORPUS_CSV)

    resultados = {}
    for clase in ["falso", "verdadero"]:
        # usamos el texto original para el análisis de POS porque spacy categoriza mejor asi
        textos = df[df["clase"] == clase]["texto_original"].tolist()
        resultados[clase] = analizar(textos)

    filas = []
    for pos in CATEGORIAS_POS:
        fila = {"POS": pos}
        for clase, etiqueta in [("verdadero", "Verdadero"), ("falso", "Falso")]:
            conteo, _, total = resultados[clase]
            fila[f"{etiqueta} (n)"] = conteo[pos]
            fila[f"{etiqueta} (%)"] = round(conteo[pos] / total * 100, 2)
        filas.append(fila)

    tabla = pd.DataFrame(filas)
    print(tabla.to_string(index=False))

    # Guardar resultados para no repetir el proceso con spaCy
    tabla.to_csv(BASE_DIR / "pos_tagging_resultado.csv", index=False)
    with open(BASE_DIR / "pos_palabras.pkl", "wb") as f:
        pickle.dump({c: resultados[c][1] for c in resultados}, f)
import spacy
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CORPUS_CSV = BASE_DIR.parent / "preprocesamiento" / "corpus_normalizado.csv"

nlp = spacy.load("es_core_news_sm")


def extraer_entidades(df: pd.DataFrame) -> pd.DataFrame:
    registros = []
    # nlp.pipe procesa en lote; zip con el resto de columnas para no perder la referencia al archivo/clase
    docs = nlp.pipe(df["texto_original"], batch_size=50)
    for (_, fila), doc in zip(df.iterrows(), docs):
        for ent in doc.ents:
            registros.append({
                "Documento": fila["archivo"],
                "Entidad": ent.text,
                "Tipo": ent.label_,
                "Clase": fila["clase"],
            })
    return pd.DataFrame(registros)


if __name__ == "__main__":
    df = pd.read_csv(CORPUS_CSV)
    tabla_entidades = extraer_entidades(df)

    print(tabla_entidades.shape)
    print(tabla_entidades.head())

    salida = BASE_DIR / "ner_entidades.csv"
    tabla_entidades.to_csv(salida, index=False, encoding="utf-8")
    print(f"Guardado en: {salida}")
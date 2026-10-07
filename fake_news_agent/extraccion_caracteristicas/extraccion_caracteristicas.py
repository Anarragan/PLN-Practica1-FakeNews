"""Punto 5 - Extracción de características (Bag of Words)."""

from __future__ import annotations

import json
import os
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.util import bigrams



# Configuración
BASE_DIR = Path(__file__).resolve().parent.parent

CORPUS_NORMALIZADO = (
    BASE_DIR
    / "preprocesamiento"
    / "corpus_normalizado.csv"
)

OUTPUT_DIR = BASE_DIR / "extraccion_caracteristicas"
VOCAB_DIR = OUTPUT_DIR / "vocabularios"

MIN_FRECUENCIA = 3

COLUMNAS_REQUERIDAS = {
    "archivo",
    "clase",
    "label",
    "texto_normalizado",
}


# Recursos y utilidades
def quitar_tildes(texto: str) -> str:
    """Quita tildes de vocales igual que el punto 2 (conserva la ñ)."""
    texto = texto.replace("ñ", "\x00").replace("Ñ", "\x01")
    nfkd = unicodedata.normalize("NFKD", texto)
    sin_tildes = "".join(c for c in nfkd if not unicodedata.combining(c))
    return sin_tildes.replace("\x00", "ñ").replace("\x01", "Ñ")


def cargar_stopwords_espanol() -> set[str]:
    """Carga y normaliza las stopwords de NLTK para español.

    El punto 2 elimina tildes del corpus, por lo que aquí se hace lo mismo con
    las stopwords. Así, por ejemplo, 'también' puede coincidir con 'tambien'.
    """
    try:
        palabras = stopwords.words("spanish")
    except LookupError:
        nltk.download("stopwords", quiet=False)
        palabras = stopwords.words("spanish")

    return {
        quitar_tildes(palabra.lower())
        for palabra in palabras
    }


# Entrada: salida del punto 2
def cargar_corpus_normalizado(
    ruta: Path = CORPUS_NORMALIZADO,
) -> pd.DataFrame:
    """Carga el corpus generado por preprocesamiento/normalizer.py."""
    if not ruta.exists():
        raise FileNotFoundError(
            "No se encontró el corpus normalizado:\n"
            f"  {ruta}\n\n"
            "Ejecuta primero el punto 2:\n"
            "  python fake_news_agent/preprocesamiento/normalizer.py"
        )

    df = pd.read_csv(ruta)

    faltantes = COLUMNAS_REQUERIDAS - set(df.columns)
    if faltantes:
        raise ValueError(
            "El corpus normalizado no tiene todas las columnas requeridas. "
            f"Faltan: {sorted(faltantes)}"
        )

    # Evita errores por valores nulos.
    df["archivo"] = df["archivo"].fillna("").astype(str)
    df["clase"] = df["clase"].fillna("").astype(str)
    df["texto_normalizado"] = (
        df["texto_normalizado"]
        .fillna("")
        .astype(str)
    )

    # Se conserva label como entero para los pasos posteriores.
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    return df



# Tokenización y Bag of Words

def tokenizar(texto_normalizado: str) -> list[str]:
    """Tokeniza un texto ya normalizado.

    Como el punto 2 ya deja el texto en minúsculas, sin signos de puntuación,
    números, enlaces ni saltos de línea, aquí basta separar por espacios.
    """
    return texto_normalizado.split()


def crear_unigramas(tokens: list[str]) -> list[str]:
    """Devuelve los tokens individuales conservando su orden."""
    return list(tokens)


def crear_bigramas(tokens: list[str]) -> list[str]:
    """Crea bigramas preservando el orden original.

    Se usa '_' para representar cada bigrama como una sola característica:
        cambio_climatico
    """
    return [f"{a}_{b}" for a, b in bigrams(tokens)]


def construir_bow(tokens: list[str]) -> list[str]:
    """Construye la bolsa de características: unigramas + bigramas."""
    return crear_unigramas(tokens) + crear_bigramas(tokens)



# Reducción por stopwords
def componentes_caracteristica(caracteristica: str) -> list[str]:
    """Separa un unigrama o bigrama en sus componentes."""
    return caracteristica.split("_")


def contiene_stopword(
    caracteristica: str,
    stop_es: set[str],
) -> bool:
    """Indica si algún componente de la característica es una stopword."""
    return any(
        componente in stop_es
        for componente in componentes_caracteristica(caracteristica)
    )


def eliminar_stopwords(
    caracteristicas: Iterable[str],
    stop_es: set[str],
) -> list[str]:
    """Elimina unigramas y bigramas asociados a stopwords."""
    return [
        caracteristica
        for caracteristica in caracteristicas
        if not contiene_stopword(caracteristica, stop_es)
    ]


# Reducción por frecuencia
def calcular_frecuencias_globales(
    documentos: Iterable[list[str]],
) -> Counter[str]:
    """Cuenta las ocurrencias de cada característica en todo el corpus."""
    frecuencias: Counter[str] = Counter()

    for caracteristicas in documentos:
        frecuencias.update(caracteristicas)

    return frecuencias


def eliminar_baja_frecuencia(
    caracteristicas: Iterable[str],
    frecuencias_globales: Counter[str],
    minimo: int = MIN_FRECUENCIA,
) -> list[str]:
    """Elimina características cuya frecuencia global sea menor que minimo."""
    return [
        caracteristica
        for caracteristica in caracteristicas
        if frecuencias_globales[caracteristica] >= minimo
    ]



# Stemming
def aplicar_stemming_caracteristica(
    caracteristica: str,
    stemmer: SnowballStemmer,
) -> str:
    """Aplica stemming a cada componente de un unigrama o bigrama."""
    componentes = componentes_caracteristica(caracteristica)
    stems = [stemmer.stem(componente) for componente in componentes]
    return "_".join(stems)


def aplicar_stemming_documento(
    caracteristicas: Iterable[str],
    stemmer: SnowballStemmer,
) -> list[str]:
    """Aplica stemming a todas las características de un documento."""
    return [
        aplicar_stemming_caracteristica(caracteristica, stemmer)
        for caracteristica in caracteristicas
    ]


# Exportación
def tipo_caracteristica(caracteristica: str) -> str:
    """Clasifica una característica como unigrama o bigrama."""
    return "bigrama" if "_" in caracteristica else "unigrama"


def guardar_vocabulario(
    ruta: Path,
    frecuencias: Counter[str],
) -> None:
    """Guarda un vocabulario ordenado de mayor a menor frecuencia."""
    registros = [
        {
            "caracteristica": caracteristica,
            "tipo": tipo_caracteristica(caracteristica),
            "frecuencia": frecuencia,
        }
        for caracteristica, frecuencia in frecuencias.most_common()
    ]

    pd.DataFrame(
        registros,
        columns=["caracteristica", "tipo", "frecuencia"],
    ).to_csv(ruta, index=False, encoding="utf-8")


def serializar_lista(elementos: list[str]) -> str:
    """Serializa una lista en JSON para almacenarla de forma segura en CSV."""
    return json.dumps(elementos, ensure_ascii=False)



# Función reutilizable del punto 5
def extraer_caracteristicas(
    df: pd.DataFrame,
    minimo_frecuencia: int = MIN_FRECUENCIA,
) -> tuple[pd.DataFrame, dict[str, Counter[str]]]:
    """Ejecuta el flujo completo del punto 5.

    Retorna
    -------
    df_resultado:
        Un DataFrame con las distintas representaciones por documento.

    frecuencias:
        Diccionario con los vocabularios/frecuencias de cada etapa.
    """
    stop_es = cargar_stopwords_espanol()
    stemmer = SnowballStemmer("spanish")

    # A. Tokenización + BoW (unigramas y bigramas)
    tokens_por_documento = [
        tokenizar(texto)
        for texto in df["texto_normalizado"]
    ]

    bow_inicial_por_documento = [
        construir_bow(tokens)
        for tokens in tokens_por_documento
    ]

    frecuencias_iniciales = calcular_frecuencias_globales(
        bow_inicial_por_documento
    )

    # B. Eliminación de stopwords
    bow_sin_stopwords_por_documento = [
        eliminar_stopwords(caracteristicas, stop_es)
        for caracteristicas in bow_inicial_por_documento
    ]

    frecuencias_sin_stopwords = calcular_frecuencias_globales(
        bow_sin_stopwords_por_documento
    )

    # C. Eliminación de términos con frecuencia global < 3
    bow_frecuencia_minima_por_documento = [
        eliminar_baja_frecuencia(
            caracteristicas,
            frecuencias_sin_stopwords,
            minimo=minimo_frecuencia,
        )
        for caracteristicas in bow_sin_stopwords_por_documento
    ]

    frecuencias_frecuencia_minima = calcular_frecuencias_globales(
        bow_frecuencia_minima_por_documento
    )

    # D. Stemming sobre el resultado reducido
    bow_stemming_por_documento = [
        aplicar_stemming_documento(caracteristicas, stemmer)
        for caracteristicas in bow_frecuencia_minima_por_documento
    ]

    frecuencias_stemming = calcular_frecuencias_globales(
        bow_stemming_por_documento
    )

    # E. Solo stemming (sin stopwords ni filtro de frecuencia).
    # Variante necesaria para Mateo en el punto 10: STOPWORDS=FALSE, STEMMING=TRUE.
    bow_solo_stemming_por_documento = [
        aplicar_stemming_documento(caracteristicas, stemmer)
        for caracteristicas in bow_inicial_por_documento
    ]

    frecuencias_solo_stemming = calcular_frecuencias_globales(
        bow_solo_stemming_por_documento
    )

    # DataFrame de salida. No se modifica el DataFrame original.
    resultado = df[
        ["archivo", "clase", "label", "texto_normalizado"]
    ].copy()

    resultado["tokens"] = [
        serializar_lista(tokens)
        for tokens in tokens_por_documento
    ]

    resultado["bow_inicial"] = [
        serializar_lista(caracteristicas)
        for caracteristicas in bow_inicial_por_documento
    ]

    resultado["bow_sin_stopwords"] = [
        serializar_lista(caracteristicas)
        for caracteristicas in bow_sin_stopwords_por_documento
    ]

    resultado["bow_frecuencia_minima_3"] = [
        serializar_lista(caracteristicas)
        for caracteristicas in bow_frecuencia_minima_por_documento
    ]

    resultado["bow_stemming"] = [
        serializar_lista(caracteristicas)
        for caracteristicas in bow_stemming_por_documento
    ]

    resultado["bow_solo_stemming"] = [
        serializar_lista(caracteristicas)
        for caracteristicas in bow_solo_stemming_por_documento
    ]

    frecuencias = {
        "01_bow_inicial": frecuencias_iniciales,
        "02_bow_sin_stopwords": frecuencias_sin_stopwords,
        "03_bow_frecuencia_minima_3": frecuencias_frecuencia_minima,
        "04_bow_stemming": frecuencias_stemming,
        "05_bow_solo_stemming": frecuencias_solo_stemming,
    }

    return resultado, frecuencias


# Flujo principal
def procesar_corpus() -> None:
    """Ejecuta y exporta exclusivamente los resultados del punto 5."""
    print("Punto 5 - Extracción de características")
    print("----------------:)----------------------")
    print(f"Leyendo: {CORPUS_NORMALIZADO}")

    df = cargar_corpus_normalizado()

    if df.empty:
        raise RuntimeError("El corpus normalizado está vacío.")

    resultado, frecuencias = extraer_caracteristicas(
        df,
        minimo_frecuencia=MIN_FRECUENCIA,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VOCAB_DIR.mkdir(parents=True, exist_ok=True)

    # Resultado por documento: será reutilizable en puntos posteriores.
    salida_documentos = OUTPUT_DIR / ""/ "corpus_caracteristicas.csv"
    resultado.to_csv(
        salida_documentos,
        index=False,
        encoding="utf-8",
    )

    # Vocabularios de cada etapa.
    for nombre_etapa, contador in frecuencias.items():
        guardar_vocabulario(
            VOCAB_DIR / f"{nombre_etapa}.csv",
            contador,
        )

    # Resumen del proceso.
    labels = df["label"].value_counts().to_dict()

    resumen = {
        "archivo_entrada": str(CORPUS_NORMALIZADO),
        "documentos_totales": int(len(df)),
        "documentos_verdaderos": int(labels.get(1, 0)),
        "documentos_falsos": int(labels.get(0, 0)),
        "frecuencia_minima": MIN_FRECUENCIA,
        "vocabulario_inicial": len(frecuencias["01_bow_inicial"]),
        "vocabulario_sin_stopwords": len(
            frecuencias["02_bow_sin_stopwords"]
        ),
        "vocabulario_frecuencia_minima_3": len(
            frecuencias["03_bow_frecuencia_minima_3"]
        ),
        "vocabulario_despues_stemming": len(
            frecuencias["04_bow_stemming"]
        ),
        "vocabulario_solo_stemming": len(
            frecuencias["05_bow_solo_stemming"]
        ),
        "archivo_salida_documentos": str(salida_documentos),
        "directorio_vocabularios": str(VOCAB_DIR),
    }

    (OUTPUT_DIR / "resumen_extraccion.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Documentos procesados : {resumen['documentos_totales']}")
    print(f"  Verdaderos          : {resumen['documentos_verdaderos']}")
    print(f"  Falsos              : {resumen['documentos_falsos']}")
    print(
        f"Vocabulario inicial   : "
        f"{resumen['vocabulario_inicial']}"
    )
    print(
        f"Sin stopwords         : "
        f"{resumen['vocabulario_sin_stopwords']}"
    )
    print(
        f"Frecuencia >= {MIN_FRECUENCIA}      : "
        f"{resumen['vocabulario_frecuencia_minima_3']}"
    )
    print(
        f"Después de stemming   : "
        f"{resumen['vocabulario_despues_stemming']}"
    )
    print()
    print(f"Resultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    # Si el equipo configuró NLTK_DATA en la terminal, se respeta esa ruta.
    nltk_data_env = os.environ.get("NLTK_DATA")

    if nltk_data_env and nltk_data_env not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_env)

    procesar_corpus()

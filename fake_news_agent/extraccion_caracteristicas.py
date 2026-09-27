"""
Punto 5 - Extracción de características (Bag of Words).

1. Lee el corpus de noticias Verdadero/Falso ya existente.
2. Tokeniza cada documento.
3. Construye unigramas y bigramas preservando el orden original.
4. Elimina unigramas y bigramas asociados a stopwords en español.
5. Elimina características cuya frecuencia global sea menor que 3.
6. Aplica stemming a las características resultantes.
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.util import bigrams


BASE_DIR = Path(__file__).resolve().parent
CORPUS_DIR = BASE_DIR / "corpus"
VERDAD_DIR = CORPUS_DIR / "Verdad"
FALSO_DIR = CORPUS_DIR / "falso"
OUTPUT_DIR = BASE_DIR / "caracteristicas"

MIN_FRECUENCIA = 3

# Tokenización simple: palabras alfabéticas en español.
# La limpieza general del texto pertenece al punto 2; aquí solo se tokeniza.
TOKEN_PATTERN = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", flags=re.UNICODE)



# Recursos NLTK
def cargar_stopwords_espanol() -> set[str]:
    """Carga las stopwords españolas, descargándolas si aún no existen."""
    try:
        return set(stopwords.words("spanish"))
    except LookupError:
        nltk.download("stopwords", quiet=False)
        return set(stopwords.words("spanish"))



# Lectura y tokenización
def leer_documentos(directorio: Path) -> list[tuple[str, str]]:
    """Retorna pares (nombre_archivo, texto) para todos los .txt del directorio."""
    if not directorio.exists():
        raise FileNotFoundError(
            f"No se encontró el directorio del corpus: {directorio}\n"
            "Verifica que el archivo esté ubicado en fake_news_agent/."
        )

    documentos: list[tuple[str, str]] = []
    for ruta in sorted(directorio.glob("*.txt")):
        texto = ruta.read_text(encoding="utf-8", errors="ignore")
        documentos.append((ruta.name, texto))
    return documentos


def tokenizar(texto: str) -> list[str]:
    """Obtiene tokens de palabras manteniendo su orden dentro del documento."""
    return TOKEN_PATTERN.findall(texto)


def construir_caracteristicas(tokens: list[str]) -> list[str]:
    """Construye unigramas y bigramas para un documento.

    Los bigramas se almacenan con un espacio entre sus dos componentes, por
    ejemplo: "cambio climatico".
    """
    unigramas = tokens
    bigramas_doc = [f"{a} {b}" for a, b in bigrams(tokens)]
    return unigramas + bigramas_doc


# Reducción por stopwords y frecuencia
def contiene_stopword(caracteristica: str, stop_es: set[str]) -> bool:
    """Indica si un unigrama/bigrama contiene alguna stopword."""
    componentes = caracteristica.split()
    return any(token.lower() in stop_es for token in componentes)


def eliminar_stopwords(
    caracteristicas: Iterable[str], stop_es: set[str]
) -> list[str]:
    """Elimina unigramas y bigramas asociados a stopwords españolas."""
    return [
        caracteristica
        for caracteristica in caracteristicas
        if not contiene_stopword(caracteristica, stop_es)
    ]


def obtener_frecuencias_globales(
    documentos: Iterable[list[str]],
) -> Counter[str]:
    """Calcula la frecuencia global de cada característica en todo el corpus."""
    contador: Counter[str] = Counter()
    for caracteristicas in documentos:
        contador.update(caracteristicas)
    return contador


def eliminar_baja_frecuencia(
    caracteristicas: Iterable[str],
    frecuencias: Counter[str],
    minimo: int = MIN_FRECUENCIA,
) -> list[str]:
    """Conserva únicamente características con frecuencia global >= minimo."""
    return [c for c in caracteristicas if frecuencias[c] >= minimo]



# Stemming
def aplicar_stemming(caracteristica: str, stemmer: SnowballStemmer) -> str:
    """Aplica stemming a un unigrama o a cada componente de un bigrama."""
    return " ".join(stemmer.stem(token.lower()) for token in caracteristica.split())


# Exportación
def guardar_vocabulario(
    ruta: Path,
    frecuencias: Counter[str],
    tipo: str,
) -> None:
    """Guarda un vocabulario ordenado por frecuencia en formato CSV."""
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.writer(archivo)
        writer.writerow(["caracteristica", "tipo", "frecuencia"])
        for caracteristica, frecuencia in frecuencias.most_common():
            writer.writerow([caracteristica, tipo, frecuencia])


def separar_por_tipo(frecuencias: Counter[str]) -> tuple[Counter[str], Counter[str]]:
    """Separa las frecuencias en unigramas y bigramas."""
    unigramas = Counter({k: v for k, v in frecuencias.items() if " " not in k})
    bigramas_ = Counter({k: v for k, v in frecuencias.items() if " " in k})
    return unigramas, bigramas_


# Flujo principal (punto 5)
def procesar_corpus() -> None:
    stop_es = cargar_stopwords_espanol()
    stemmer = SnowballStemmer("spanish")

    documentos_etiquetados: list[tuple[str, str, str]] = []

    for nombre, texto in leer_documentos(VERDAD_DIR):
        documentos_etiquetados.append((nombre, "Verdad", texto))

    for nombre, texto in leer_documentos(FALSO_DIR):
        documentos_etiquetados.append((nombre, "Falso", texto))

    if not documentos_etiquetados:
        raise RuntimeError("El corpus no contiene archivos .txt para procesar.")

    # Etapa A: tokenización + unigramas + bigramas.
    caracteristicas_por_documento: list[list[str]] = []
    for _, _, texto in documentos_etiquetados:
        tokens = tokenizar(texto)
        caracteristicas_por_documento.append(construir_caracteristicas(tokens))

    frecuencias_iniciales = obtener_frecuencias_globales(caracteristicas_por_documento)

    # Etapa B: reducción por stopwords.
    sin_stopwords_por_documento = [
        eliminar_stopwords(caracteristicas, stop_es)
        for caracteristicas in caracteristicas_por_documento
    ]
    frecuencias_sin_stopwords = obtener_frecuencias_globales(sin_stopwords_por_documento)

    # Etapa C: reducción por frecuencia global (< 3 se elimina).
    reducidas_por_documento = [
        eliminar_baja_frecuencia(
            caracteristicas,
            frecuencias_sin_stopwords,
            minimo=MIN_FRECUENCIA,
        )
        for caracteristicas in sin_stopwords_por_documento
    ]
    frecuencias_reducidas = obtener_frecuencias_globales(reducidas_por_documento)

    # Etapa D: stemming sobre las características resultantes.
    stem_por_documento = [
        [aplicar_stemming(c, stemmer) for c in caracteristicas]
        for caracteristicas in reducidas_por_documento
    ]
    frecuencias_stem = obtener_frecuencias_globales(stem_por_documento)

    # Resultados del punto 5.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for nombre_etapa, frecuencias in (
        ("01_bow_inicial", frecuencias_iniciales),
        ("02_bow_sin_stopwords", frecuencias_sin_stopwords),
        ("03_bow_frecuencia_minima_3", frecuencias_reducidas),
        ("04_bow_stemming", frecuencias_stem),
    ):
        unigramas, bigramas_ = separar_por_tipo(frecuencias)
        guardar_vocabulario(
            OUTPUT_DIR / f"{nombre_etapa}_unigramas.csv",
            unigramas,
            "unigrama",
        )
        guardar_vocabulario(
            OUTPUT_DIR / f"{nombre_etapa}_bigramas.csv",
            bigramas_,
            "bigrama",
        )

    resumen = {
        "documentos_totales": len(documentos_etiquetados),
        "documentos_verdad": sum(1 for _, clase, _ in documentos_etiquetados if clase == "Verdad"),
        "documentos_falso": sum(1 for _, clase, _ in documentos_etiquetados if clase == "Falso"),
        "frecuencia_minima": MIN_FRECUENCIA,
        "vocabulario_inicial": len(frecuencias_iniciales),
        "vocabulario_sin_stopwords": len(frecuencias_sin_stopwords),
        "vocabulario_frecuencia_minima_3": len(frecuencias_reducidas),
        "vocabulario_despues_stemming": len(frecuencias_stem),
    }

    (OUTPUT_DIR / "resumen_extraccion.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nPunto 5 - Extracción de características completado")
    print("--------------------------------------------------")
    print(f"Documentos procesados: {resumen['documentos_totales']}")
    print(f"  Verdad: {resumen['documentos_verdad']}")
    print(f"  Falso : {resumen['documentos_falso']}")
    print(f"Características iniciales: {resumen['vocabulario_inicial']}")
    print(f"Después de stopwords: {resumen['vocabulario_sin_stopwords']}")
    print(
        "Después de frecuencia mínima 3: "
        f"{resumen['vocabulario_frecuencia_minima_3']}"
    )
    print(f"Después de stemming: {resumen['vocabulario_despues_stemming']}")
    print(f"\nResultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    # Respeta NLTK_DATA si los compas ya lo configuraron desde el README.
    nltk_data_env = os.environ.get("NLTK_DATA")
    if nltk_data_env and nltk_data_env not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_env)

    procesar_corpus()

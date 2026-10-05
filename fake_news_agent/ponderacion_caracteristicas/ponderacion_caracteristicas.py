"""Punto 6 - Ponderación de las características.
Las matrices completas se almacenan en formato sparse .npz para evitar
crear archivos CSV enormes y consumir memoria innecesariamente."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.sparse import save_npz
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
except ImportError as exc:
    raise ImportError(
        "Faltan dependencias para ejecutar el punto 6.\n"
        "Agrega 'scikit-learn' al archivo requirements.txt y ejecuta:\n"
        "    pip install -r requirements.txt\n"
        "scikit-learn instalará también las dependencias necesarias."
    ) from exc


# Configuración de rutas

# fake_news_agent/ponderacion_caracteristicas/
CARPETA_ACTUAL = Path(__file__).resolve().parent

# Carpeta:
# fake_news_agent/
FAKE_NEWS_AGENT_DIR = CARPETA_ACTUAL.parent

# Salida del punto 5.
CORPUS_CARACTERISTICAS = (
    FAKE_NEWS_AGENT_DIR
    / "extraccion_caracteristicas"
    / "corpus_caracteristicas.csv"
)

OUTPUT_DIR = CARPETA_ACTUAL

# Configuración de la representación que se ponderará

# Esta es la salida final del punto 5:
# stopwords + frecuencia mínima 3 + stemming.
COLUMNA_CARACTERISTICAS = "bow_stemming"

COLUMNAS_METADATA = [
    "archivo",
    "clase",
    "label",
]

# Cantidad de documentos y características que se muestran en el CSV
# pequeño de inspección. Las matrices completas quedan en .npz.
N_DOCUMENTOS_MUESTRA = 10
N_CARACTERISTICAS_MUESTRA = 20


# Carga de la salida del punto 5
def cargar_corpus_caracteristicas(
    ruta: Path = CORPUS_CARACTERISTICAS,
) -> pd.DataFrame:
    """Carga y valida el archivo generado por el punto 5."""
    if not ruta.exists():
        raise FileNotFoundError(
            "No se encontró la salida del punto 5:\n"
            f"  {ruta}\n\n"
            "Ejecuta primero:\n"
            "  python "
            "fake_news_agent/extraccion_caracteristicas/"
            "extraccion_caracteristicas.py"
        )

    df = pd.read_csv(ruta)

    requeridas = set(COLUMNAS_METADATA + [COLUMNA_CARACTERISTICAS])
    faltantes = requeridas - set(df.columns)

    if faltantes:
        raise ValueError(
            "El archivo corpus_caracteristicas.csv no tiene todas las "
            "columnas requeridas. "
            f"Faltan: {sorted(faltantes)}"
        )

    if df.empty:
        raise RuntimeError(
            "corpus_caracteristicas.csv está vacío."
        )

    df["archivo"] = df["archivo"].fillna("").astype(str)
    df["clase"] = df["clase"].fillna("").astype(str)
    df["label"] = pd.to_numeric(
        df["label"],
        errors="raise",
    ).astype(int)

    return df


def deserializar_caracteristicas(valor: object) -> list[str]:
    """Convierte la lista JSON guardada por el punto 5 a list[str]."""
    if pd.isna(valor):
        return []

    if isinstance(valor, list):
        return [str(elemento) for elemento in valor]

    texto = str(valor).strip()

    if not texto:
        return []

    try:
        elementos = json.loads(texto)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "No fue posible interpretar una lista de características "
            "del archivo del punto 5. "
            f"Valor problemático: {texto[:120]!r}"
        ) from exc

    if not isinstance(elementos, list):
        raise ValueError(
            "Cada celda de características debe contener una lista JSON."
        )

    return [str(elemento) for elemento in elementos]


# Ponderación
def construir_matriz_to(
    documentos_caracteristicas: list[list[str]],
):
    """Construye la matriz de frecuencia absoluta (TO).

    La entrada ya contiene las características extraídas en el punto 5,
    por lo que CountVectorizer no vuelve a tokenizar ni preprocesar.
    Cada elemento de la lista se interpreta directamente como una
    característica.
    """
    vectorizador = CountVectorizer(
        analyzer=lambda documento: documento,
        lowercase=False,
        token_pattern=None,
        dtype=np.int64,
    )

    matriz_to = vectorizador.fit_transform(
        documentos_caracteristicas
    )

    caracteristicas = vectorizador.get_feature_names_out()

    return matriz_to, caracteristicas


def construir_matriz_tfidf(matriz_to):
    """Calcula TF-IDF a partir de la matriz TO.

    Se utiliza:
        TF  = frecuencia absoluta de la característica en el documento.
        IDF = inversa de la frecuencia documental con suavizado.

    norm=None evita agregar una normalización L1/L2 adicional que no es
    solicitada explícitamente por el numeral 6.
    """
    transformador = TfidfTransformer(
        norm=None,
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    )

    matriz_tfidf = transformador.fit_transform(matriz_to)

    return matriz_tfidf, transformador.idf_


# Resultados auxiliares
def construir_tabla_vocabulario(
    caracteristicas: np.ndarray,
    matriz_to,
    matriz_tfidf,
    idf: np.ndarray,
) -> pd.DataFrame:
    """Crea una tabla interpretable con información de cada característica."""
    frecuencia_total_to = np.asarray(
        matriz_to.sum(axis=0)
    ).ravel()

    frecuencia_documental = np.asarray(
        (matriz_to > 0).sum(axis=0)
    ).ravel()

    suma_tfidf = np.asarray(
        matriz_tfidf.sum(axis=0)
    ).ravel()

    return pd.DataFrame(
        {
            "indice": np.arange(len(caracteristicas)),
            "caracteristica": caracteristicas,
            "tipo": [
                "bigrama" if "_" in caracteristica else "unigrama"
                for caracteristica in caracteristicas
            ],
            "frecuencia_total_TO": frecuencia_total_to.astype(int),
            "documentos_con_caracteristica": (
                frecuencia_documental.astype(int)
            ),
            "IDF": idf,
            "suma_TFIDF_corpus": suma_tfidf,
        }
    )


def guardar_muestra_ponderaciones(
    matriz_to,
    matriz_tfidf,
    caracteristicas: np.ndarray,
    metadata: pd.DataFrame,
    ruta: Path,
) -> None:
    """Guarda una muestra pequeña legible de ambas ponderaciones.

    No reemplaza las matrices completas .npz; solo permite verificar
    visualmente que la ponderación se generó correctamente.
    """
    n_docs = min(N_DOCUMENTOS_MUESTRA, matriz_to.shape[0])

    # Se eligen las características con mayor frecuencia TO global para
    # que la muestra sea informativa y no columnas arbitrarias.
    totales = np.asarray(matriz_to.sum(axis=0)).ravel()
    n_features = min(
        N_CARACTERISTICAS_MUESTRA,
        matriz_to.shape[1],
    )

    indices_top = np.argsort(totales)[::-1][:n_features]

    registros: list[dict[str, object]] = []

    for indice_documento in range(n_docs):
        for indice_feature in indices_top:
            valor_to = matriz_to[
                indice_documento,
                indice_feature,
            ]

            valor_tfidf = matriz_tfidf[
                indice_documento,
                indice_feature,
            ]

            registros.append(
                {
                    "archivo": metadata.iloc[
                        indice_documento
                    ]["archivo"],
                    "label": int(
                        metadata.iloc[indice_documento]["label"]
                    ),
                    "caracteristica": caracteristicas[
                        indice_feature
                    ],
                    "TO": int(valor_to),
                    "TF_IDF": float(valor_tfidf),
                }
            )

    pd.DataFrame(registros).to_csv(
        ruta,
        index=False,
        encoding="utf-8",
    )


# Flujo principal
def ponderar_caracteristicas() -> None:
    """Ejecuta exclusivamente el numeral 6 de la práctica."""
    print("Punto 6 - Ponderación de las características")
    print("---------------------:)-----------------------")
    print(f"Entrada: {CORPUS_CARACTERISTICAS}")
    print(
        "Representación usada: "
        f"{COLUMNA_CARACTERISTICAS}"
    )

    df = cargar_corpus_caracteristicas()

    documentos_caracteristicas = [
        deserializar_caracteristicas(valor)
        for valor in df[COLUMNA_CARACTERISTICAS]
    ]

    if not any(documentos_caracteristicas):
        raise RuntimeError(
            "La columna seleccionada no contiene características."
        )

    # 1. Frecuencia absoluta / Term Occurrences (TO).
    matriz_to, caracteristicas = construir_matriz_to(
        documentos_caracteristicas
    )

    # 2. TF-IDF.
    matriz_tfidf, idf = construir_matriz_tfidf(
        matriz_to
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Matrices completas en formato sparse.
    ruta_to = OUTPUT_DIR / "matriz_to.npz"
    ruta_tfidf = OUTPUT_DIR / "matriz_tfidf.npz"

    save_npz(ruta_to, matriz_to)
    save_npz(ruta_tfidf, matriz_tfidf)

    # Metadatos/etiquetas en el mismo orden de las filas de las matrices.
    metadata = df[COLUMNAS_METADATA].copy()

    ruta_documentos = OUTPUT_DIR / "documentos.csv"
    metadata.to_csv(
        ruta_documentos,
        index=False,
        encoding="utf-8",
    )

    # Relación índice de columna <-> característica y datos descriptivos.
    tabla_vocabulario = construir_tabla_vocabulario(
        caracteristicas,
        matriz_to,
        matriz_tfidf,
        idf,
    )

    ruta_vocabulario = OUTPUT_DIR / "vocabulario_ponderacion.csv"
    tabla_vocabulario.to_csv(
        ruta_vocabulario,
        index=False,
        encoding="utf-8",
    )

    # Muestra legible para inspección manual.
    ruta_muestra = OUTPUT_DIR / "muestra_ponderaciones.csv"
    guardar_muestra_ponderaciones(
        matriz_to,
        matriz_tfidf,
        caracteristicas,
        metadata,
        ruta_muestra,
    )

    # Resumen.
    labels = metadata["label"].value_counts().to_dict()

    resumen = {
        "archivo_entrada": str(CORPUS_CARACTERISTICAS),
        "columna_caracteristicas": COLUMNA_CARACTERISTICAS,
        "documentos": int(matriz_to.shape[0]),
        "caracteristicas": int(matriz_to.shape[1]),
        "documentos_verdaderos": int(labels.get(1, 0)),
        "documentos_falsos": int(labels.get(0, 0)),
        "ponderaciones_generadas": [
            "TO",
            "TF-IDF",
        ],
        "tfidf": {
            "tf": "frecuencia absoluta",
            "idf": "log((1 + n_documentos) / "
            "(1 + frecuencia_documental)) + 1",
            "smooth_idf": True,
            "normalizacion_final": None,
        },
        "archivos_salida": {
            "matriz_to": str(ruta_to),
            "matriz_tfidf": str(ruta_tfidf),
            "documentos": str(ruta_documentos),
            "vocabulario": str(ruta_vocabulario),
            "muestra": str(ruta_muestra),
        },
    }

    ruta_resumen = OUTPUT_DIR / "resumen_ponderacion.json"
    ruta_resumen.write_text(
        json.dumps(
            resumen,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Ponderación completada")
    print("----------------------")
    print(
        f"Documentos       : {matriz_to.shape[0]}"
    )
    print(
        f"Características  : {matriz_to.shape[1]}"
    )
    print(
        f"Valores TO != 0  : {matriz_to.nnz}"
    )
    print(
        f"Valores TF-IDF != 0: {matriz_tfidf.nnz}"
    )
    print()
    print(f"Resultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    ponderar_caracteristicas()

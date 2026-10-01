"""Punto 8 - Análisis de temas según la veracidad de las noticias."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from gensim.corpora import Dictionary
from gensim.models import LdaModel



# Rutas
CARPETA_ACTUAL = Path(__file__).resolve().parent
FAKE_NEWS_AGENT_DIR = CARPETA_ACTUAL.parent

CORPUS_CARACTERISTICAS = (
    FAKE_NEWS_AGENT_DIR
    / "extraccion_caracteristicas"
    / "corpus_caracteristicas.csv"
)

MATRIZ_TFIDF = (
    FAKE_NEWS_AGENT_DIR
    / "ponderacion_caracteristicas"
    / "matriz_tfidf.npz"
)

MODELO_TEMAS_DIR = (
    FAKE_NEWS_AGENT_DIR
    / "modelado_temas"
)

RESUMEN_MODELADO = (
    MODELO_TEMAS_DIR
    / "resumen_modelado_temas.json"
)

TEMAS_LSA = MODELO_TEMAS_DIR / "temas_lsa.csv"
TEMAS_LDA = MODELO_TEMAS_DIR / "temas_lda.csv"

MODELO_LSA = (
    MODELO_TEMAS_DIR
    / "modelos"
    / "modelo_lsa.joblib"
)

MODELO_LDA = (
    MODELO_TEMAS_DIR
    / "modelos"
    / "modelo_lda.gensim"
)

DICCIONARIO_GENSIM = (
    MODELO_TEMAS_DIR
    / "modelos"
    / "diccionario_gensim.dict"
)

OUTPUT_DIR = CARPETA_ACTUAL



# Configuración

# "auto", "lda" o "lsa"
MODELO_SELECCIONADO = "lda"

# Salida final del punto 5
COLUMNA_BOW = "bow_stemming"



# Carga y selección del modelo
def validar_archivos(rutas: list[Path]) -> None:
    faltantes = [ruta for ruta in rutas if not ruta.exists()]

    if faltantes:
        detalle = "\n".join(f"  - {ruta}" for ruta in faltantes)
        raise FileNotFoundError(
            "Faltan archivos necesarios para ejecutar el punto 8:\n"
            f"{detalle}\n\n"
            "Ejecuta primero los puntos 5, 6 y 7."
        )


def cargar_resumen_modelado() -> dict:
    validar_archivos([RESUMEN_MODELADO])
    return json.loads(
        RESUMEN_MODELADO.read_text(encoding="utf-8")
    )


def seleccionar_modelo(resumen: dict) -> tuple[str, str]:
    opcion = MODELO_SELECCIONADO.lower().strip()

    if opcion in {"lda", "lsa"}:
        return opcion, f"Selección manual: {opcion.upper()}"

    if opcion != "auto":
        raise ValueError(
            "MODELO_SELECCIONADO debe ser 'auto', 'lda' o 'lsa'."
        )

    coherencia_lsa = float(resumen["lsa"]["coherencia"])
    coherencia_lda = float(resumen["lda"]["coherencia"])

    if coherencia_lda >= coherencia_lsa:
        return (
            "lda",
            (
                "Mayor Score de Coherencia c_v. "
                f"LDA={coherencia_lda:.6f}, "
                f"LSA={coherencia_lsa:.6f}"
            ),
        )

    return (
        "lsa",
        (
            "Mayor Score de Coherencia c_v. "
            f"LSA={coherencia_lsa:.6f}, "
            f"LDA={coherencia_lda:.6f}"
        ),
    )


def cargar_corpus() -> pd.DataFrame:
    validar_archivos([CORPUS_CARACTERISTICAS])

    df = pd.read_csv(CORPUS_CARACTERISTICAS)

    requeridas = {
        "archivo",
        "clase",
        "label",
        COLUMNA_BOW,
    }

    faltantes = requeridas - set(df.columns)

    if faltantes:
        raise ValueError(
            "Faltan columnas en corpus_caracteristicas.csv: "
            f"{sorted(faltantes)}"
        )

    df["archivo"] = df["archivo"].fillna("").astype(str)
    df["clase"] = df["clase"].fillna("").astype(str)
    df["label"] = pd.to_numeric(
        df["label"],
        errors="raise",
    ).astype(int)

    return df


def deserializar_caracteristicas(valor: object) -> list[str]:
    if pd.isna(valor):
        return []

    if isinstance(valor, list):
        return [str(x) for x in valor]

    texto = str(valor).strip()

    if not texto:
        return []

    elementos = json.loads(texto)

    if not isinstance(elementos, list):
        raise ValueError(
            f"La columna {COLUMNA_BOW!r} debe contener listas JSON."
        )

    return [str(x) for x in elementos]


# Tema dominante con LDA
def temas_dominantes_lda(df: pd.DataFrame) -> pd.DataFrame:
    validar_archivos([MODELO_LDA, DICCIONARIO_GENSIM])

    modelo = LdaModel.load(str(MODELO_LDA))
    diccionario = Dictionary.load(str(DICCIONARIO_GENSIM))

    resultados = []

    for valor in df[COLUMNA_BOW]:
        tokens = deserializar_caracteristicas(valor)
        bow = diccionario.doc2bow(tokens)

        distribucion = modelo.get_document_topics(
            bow,
            minimum_probability=0.0,
        )

        indice_tema, probabilidad = max(
            distribucion,
            key=lambda par: par[1],
        )

        resultados.append(
            {
                "indice_tema": int(indice_tema) + 1,
                "tema_dominante": f"Tema {int(indice_tema) + 1}",
                "fuerza_tema": float(probabilidad),
            }
        )

    return pd.DataFrame(resultados)


# Tema dominante con LSA
def temas_dominantes_lsa(df: pd.DataFrame) -> pd.DataFrame:
    validar_archivos([MODELO_LSA, MATRIZ_TFIDF])

    modelo = joblib.load(MODELO_LSA)
    matriz_tfidf = load_npz(MATRIZ_TFIDF)

    if matriz_tfidf.shape[0] != len(df):
        raise ValueError(
            "La matriz TF-IDF y el corpus no tienen el mismo número de documentos."
        )

    representacion = modelo.transform(matriz_tfidf)

    indices = np.argmax(np.abs(representacion), axis=1)
    fuerzas = np.max(np.abs(representacion), axis=1)

    return pd.DataFrame(
        {
            "indice_tema": indices.astype(int) + 1,
            "tema_dominante": [
                f"Tema {indice + 1}"
                for indice in indices
            ],
            "fuerza_tema": fuerzas.astype(float),
        }
    )

# Descripción de temas
def cargar_descripcion_temas(modelo: str) -> pd.DataFrame:
    if modelo == "lda":
        validar_archivos([TEMAS_LDA])
        temas = pd.read_csv(TEMAS_LDA)

        if "interpretacion" not in temas.columns:
            temas["interpretacion"] = ""

    else:
        validar_archivos([TEMAS_LSA])
        temas = pd.read_csv(TEMAS_LSA)

        if "interpretacion_manual" in temas.columns:
            temas["interpretacion"] = (
                temas["interpretacion_manual"]
                .fillna("")
                .astype(str)
            )
        else:
            temas["interpretacion"] = ""

    return temas[
        [
            "tema",
            "palabras_representativas",
            "interpretacion",
        ]
    ].copy()



# Distribución por veracidad
def construir_distribucion(
    documentos: pd.DataFrame,
    descripcion_temas: pd.DataFrame,
) -> pd.DataFrame:

    total_verdaderas = int(
        (documentos["label"] == 1).sum()
    )
    total_falsas = int(
        (documentos["label"] == 0).sum()
    )

    if total_verdaderas == 0 or total_falsas == 0:
        raise RuntimeError(
            "Se requieren documentos de ambas clases."
        )

    filas = []

    for _, fila_tema in descripcion_temas.iterrows():
        tema = str(fila_tema["tema"])

        verdaderas = int(
            (
                (documentos["label"] == 1)
                & (documentos["tema_dominante"] == tema)
            ).sum()
        )

        falsas = int(
            (
                (documentos["label"] == 0)
                & (documentos["tema_dominante"] == tema)
            ).sum()
        )

        porcentaje_verdaderas = (
            verdaderas / total_verdaderas * 100
        )
        porcentaje_falsas = (
            falsas / total_falsas * 100
        )

        filas.append(
            {
                "tema": tema,
                "palabras_representativas": (
                    fila_tema["palabras_representativas"]
                ),
                "verdaderas_cantidad": verdaderas,
                "verdaderas_porcentaje": porcentaje_verdaderas,
                "falsas_cantidad": falsas,
                "falsas_porcentaje": porcentaje_falsas,
                "diferencia_puntos_porcentuales": (
                    porcentaje_verdaderas
                    - porcentaje_falsas
                ),
                "diferencia_absoluta": abs(
                    porcentaje_verdaderas
                    - porcentaje_falsas
                ),
                "interpretacion": fila_tema["interpretacion"],
            }
        )

    return pd.DataFrame(filas)


def generar_grafica(
    distribucion: pd.DataFrame,
    ruta: Path,
) -> None:

    x = np.arange(len(distribucion))
    ancho = 0.38

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        x - ancho / 2,
        distribucion["verdaderas_porcentaje"],
        width=ancho,
        label="Verdaderas",
    )

    ax.bar(
        x + ancho / 2,
        distribucion["falsas_porcentaje"],
        width=ancho,
        label="Falsas",
    )

    ax.set_xlabel("Tema")
    ax.set_ylabel("Documentos (%)")
    ax.set_title(
        "Distribución porcentual de temas: verdaderas vs. falsas"
    )
    ax.set_xticks(x, distribucion["tema"])
    ax.legend()

    fig.tight_layout()
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generar_analisis_textual(
    distribucion: pd.DataFrame,
    modelo: str,
    criterio: str,
) -> str:

    ordenada = distribucion.sort_values(
        "diferencia_absoluta",
        ascending=False,
    )

    lineas = [
        "ANÁLISIS DE TEMAS SEGÚN LA VERACIDAD",
        "=" * 42,
        "",
        f"Modelo utilizado: {modelo.upper()}",
        f"Criterio de selección: {criterio}",
        "",
        "Temas con mayor diferencia proporcional:",
    ]

    for _, fila in ordenada.head(5).iterrows():
        diferencia = float(
            fila["diferencia_puntos_porcentuales"]
        )

        if diferencia > 0:
            predominio = "Verdaderas"
        elif diferencia < 0:
            predominio = "Falsas"
        else:
            predominio = "Sin diferencia"

        lineas.append(
            (
                f"- {fila['tema']}: "
                f"Verdaderas={fila['verdaderas_porcentaje']:.2f}%, "
                f"Falsas={fila['falsas_porcentaje']:.2f}%, "
                f"diferencia={abs(diferencia):.2f} pp. "
                f"Mayor presencia: {predominio}."
            )
        )

    return "\n".join(lineas)



# Flujo principal
def analizar_temas_veracidad() -> None:
    print(
        "Punto 8 - Análisis de temas según la veracidad de las noticias"
    )
    print(
        "--------------------------------------------------------------"
    )

    resumen_modelado = cargar_resumen_modelado()

    modelo, criterio = seleccionar_modelo(
        resumen_modelado
    )

    print(f"Modelo seleccionado: {modelo.upper()}")
    print(f"Criterio: {criterio}")
    print()

    df = cargar_corpus()

    if modelo == "lda":
        resultado_temas = temas_dominantes_lda(df)
    else:
        resultado_temas = temas_dominantes_lsa(df)

    documentos = df[
        ["archivo", "clase", "label"]
    ].copy()

    documentos = pd.concat(
        [
            documentos.reset_index(drop=True),
            resultado_temas.reset_index(drop=True),
        ],
        axis=1,
    )

    descripcion_temas = cargar_descripcion_temas(
        modelo
    )

    distribucion = construir_distribucion(
        documentos,
        descripcion_temas,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Tema dominante por documento
    ruta_documentos = (
        OUTPUT_DIR / "temas_por_documento.csv"
    )
    documentos.to_csv(
        ruta_documentos,
        index=False,
        encoding="utf-8",
    )

    # Tabla solicitada en el PDF
    ruta_tabla = (
        OUTPUT_DIR / "tabla_distribucion_temas.csv"
    )
    distribucion.to_csv(
        ruta_tabla,
        index=False,
        encoding="utf-8",
    )

    # Gráfico comparativo
    ruta_grafica = (
        OUTPUT_DIR / "grafico_distribucion_temas.png"
    )
    generar_grafica(
        distribucion,
        ruta_grafica,
    )

    # Diferencias ordenadas
    diferencias = distribucion.sort_values(
        "diferencia_absoluta",
        ascending=False,
    )

    ruta_diferencias = (
        OUTPUT_DIR / "diferencias_temas.csv"
    )
    diferencias.to_csv(
        ruta_diferencias,
        index=False,
        encoding="utf-8",
    )

    # Apoyo textual
    analisis = generar_analisis_textual(
        distribucion,
        modelo,
        criterio,
    )

    ruta_analisis = (
        OUTPUT_DIR / "analisis_diferencias.txt"
    )
    ruta_analisis.write_text(
        analisis,
        encoding="utf-8",
    )

    # Resumen
    mayor = diferencias.iloc[0]

    resumen = {
        "modelo_utilizado": modelo.upper(),
        "criterio_seleccion_modelo": criterio,
        "documentos_totales": int(len(documentos)),
        "documentos_verdaderos": int(
            (documentos["label"] == 1).sum()
        ),
        "documentos_falsos": int(
            (documentos["label"] == 0).sum()
        ),
        "numero_temas": int(len(distribucion)),
        "criterio_tema_dominante": (
            "Mayor probabilidad de tema"
            if modelo == "lda"
            else "Mayor magnitud absoluta del componente LSA"
        ),
        "tema_mayor_diferencia": {
            "tema": str(mayor["tema"]),
            "verdaderas_porcentaje": float(
                mayor["verdaderas_porcentaje"]
            ),
            "falsas_porcentaje": float(
                mayor["falsas_porcentaje"]
            ),
            "diferencia_absoluta": float(
                mayor["diferencia_absoluta"]
            ),
        }
    }

    ruta_resumen = (
        OUTPUT_DIR / "resumen_analisis_temas.json"
    )
    ruta_resumen.write_text(
        json.dumps(
            resumen,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Análisis completado")
    print("-------------------")
    print(f"Documentos: {len(documentos)}")
    print(f"Temas analizados: {len(distribucion)}")
    print(
        "Mayor diferencia proporcional: "
        f"{mayor['tema']} "
        f"({mayor['diferencia_absoluta']:.2f} pp)"
    )
    print()
    print(f"Resultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    analizar_temas_veracidad()

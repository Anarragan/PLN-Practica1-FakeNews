from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

try:
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel, LdaModel
except ImportError as exc:
    raise ImportError(
        "Falta Gensim para ejecutar el punto 7.\n"
        "Añade 'gensim' al requirements.txt y ejecuta:\n"
        "    pip install -r requirements.txt"
    ) from exc



# Configuración de rutas

# fake_news_agent/modelado_temas/
CARPETA_ACTUAL = Path(__file__).resolve().parent

# fake_news_agent/
FAKE_NEWS_AGENT_DIR = CARPETA_ACTUAL.parent

# Salida del punto 5: contiene el Bag of Words final por documento.
CORPUS_CARACTERISTICAS = (
    FAKE_NEWS_AGENT_DIR
    / "extraccion_caracteristicas"
    / "corpus_caracteristicas.csv"
)

# Salidas del punto 6: TF-IDF para LSA y vocabulario asociado.
MATRIZ_TFIDF = (
    FAKE_NEWS_AGENT_DIR
    / "ponderacion_caracteristicas"
    / "matriz_tfidf.npz"
)

VOCABULARIO_PONDERACION = (
    FAKE_NEWS_AGENT_DIR
    / "ponderacion_caracteristicas"
    / "vocabulario_ponderacion.csv"
)

# Todo lo generado por el punto 7 queda en su propia carpeta.
OUTPUT_DIR = CARPETA_ACTUAL
MODELOS_DIR = OUTPUT_DIR / "modelos"


# Parámetros del punto 7

# Salida final del punto 5:
# stopwords + frecuencia >= 3 + stemming.
COLUMNA_BOW = "bow_stemming"

# k = 2, 3, ..., 10.
VALORES_K = list(range(2, 11))

# 10 palabras/características más representativas.
TOP_N = 10

RANDOM_STATE = 42

# Parámetros de entrenamiento de LDA.
# Pueden aumentarse.
LDA_PASSES = 20
LDA_ITERATIONS = 100

# Interpretación manual de los temas (corrida final: LSA k=2, LDA k=4).
INTERPRETACIONES_LSA = {
    "Tema 1": (
        "Casi todas las palabras vienen de la frase 'Iniciativa vers per "
        "Catalunya', que solo aparece en las noticias falsas de la fuente 2. "
        "Más que un tema, es un error que quedó en el dataset."
    ),
    "Tema 2": (
        "Separa dos grupos: de un lado las noticias que tienen la frase de "
        "'Iniciativa vers per Catalunya' y del otro las que hablan del PP, "
        "el Gobierno y Podemos."
    ),
}

INTERPRETACIONES_LDA = {
    "Tema 1": (
        "Noticias sobre la oposición en el Congreso (PP, Vox, Podemos) y "
        "sobre la Guardia Civil en el juicio del procés."
    ),
    "Tema 2": (
        "información y publicaciones digitales sobre Argentina, posiblemente con referencias a Buenos Aires."
    ),
    "Tema 3": (
        "Casos judiciales: condenas del Supremo, denuncias y exhumaciones "
        "de víctimas del franquismo."
    ),
    "Tema 4": (
        "Las negociaciones del Gobierno de Pedro Sánchez con ERC y los "
        "independentistas para sacar adelante presupuestos y votaciones."
    )
}


# Carga y validación de entradas
def cargar_entradas():
    """Carga y valida las salidas de los puntos 5 y 6."""
    archivos_requeridos = [
        CORPUS_CARACTERISTICAS,
        MATRIZ_TFIDF,
        VOCABULARIO_PONDERACION,
    ]

    faltantes = [
        ruta
        for ruta in archivos_requeridos
        if not ruta.exists()
    ]

    if faltantes:
        rutas = "\n".join(f"  - {ruta}" for ruta in faltantes)
        raise FileNotFoundError(
            "Faltan archivos necesarios para ejecutar el punto 7:\n"
            f"{rutas}\n\n"
            "Ejecuta primero los puntos 5 y 6."
        )

    df = pd.read_csv(CORPUS_CARACTERISTICAS)

    columnas_requeridas = {
        "archivo",
        "clase",
        "label",
        COLUMNA_BOW,
    }

    columnas_faltantes = columnas_requeridas - set(df.columns)

    if columnas_faltantes:
        raise ValueError(
            "corpus_caracteristicas.csv no contiene todas las columnas "
            f"requeridas. Faltan: {sorted(columnas_faltantes)}"
        )

    if df.empty:
        raise RuntimeError(
            "corpus_caracteristicas.csv está vacío."
        )

    matriz_tfidf = load_npz(MATRIZ_TFIDF)

    vocabulario_df = pd.read_csv(VOCABULARIO_PONDERACION)

    columnas_vocabulario = {
        "indice",
        "caracteristica",
    }

    faltantes_vocabulario = (
        columnas_vocabulario - set(vocabulario_df.columns)
    )

    if faltantes_vocabulario:
        raise ValueError(
            "vocabulario_ponderacion.csv no tiene las columnas necesarias. "
            f"Faltan: {sorted(faltantes_vocabulario)}"
        )

    # Se ordena por índice para garantizar que cada palabra coincida con la
    # misma columna de matriz_tfidf.npz.
    vocabulario_df = (
        vocabulario_df
        .sort_values("indice")
        .reset_index(drop=True)
    )

    vocabulario = (
        vocabulario_df["caracteristica"]
        .astype(str)
        .to_numpy()
    )

    if matriz_tfidf.shape[0] != len(df):
        raise ValueError(
            "La cantidad de documentos de matriz_tfidf.npz no coincide "
            "con corpus_caracteristicas.csv.\n"
            f"Matriz TF-IDF: {matriz_tfidf.shape[0]} documentos\n"
            f"Corpus: {len(df)} documentos"
        )

    if matriz_tfidf.shape[1] != len(vocabulario):
        raise ValueError(
            "La cantidad de columnas de matriz_tfidf.npz no coincide "
            "con vocabulario_ponderacion.csv.\n"
            f"Matriz TF-IDF: {matriz_tfidf.shape[1]} características\n"
            f"Vocabulario: {len(vocabulario)} características"
        )

    return df, matriz_tfidf, vocabulario


def deserializar_caracteristicas(valor: object) -> list[str]:
    """Convierte la lista JSON guardada en el punto 5 a list[str]."""
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
            "No fue posible leer una lista de características del punto 5. "
            f"Valor problemático: {texto[:120]!r}"
        ) from exc

    if not isinstance(elementos, list):
        raise ValueError(
            "Cada celda de bow_stemming debe contener una lista JSON."
        )

    return [str(elemento) for elemento in elementos]



# Corpus Gensim para coherencia y LDA
def preparar_corpus_gensim(
    documentos_tokens: list[list[str]],
):
    """Construye Dictionary y corpus Bag of Words de Gensim.

    No se aplica ningún filtro adicional, porque la reducción por stopwords,
    frecuencia mínima y stemming ya se realizó en el punto 5.
    """
    diccionario = Dictionary(documentos_tokens)

    corpus_bow = [
        diccionario.doc2bow(documento)
        for documento in documentos_tokens
    ]

    if len(diccionario) == 0:
        raise RuntimeError(
            "El vocabulario de Gensim quedó vacío."
        )

    return diccionario, corpus_bow



# LSA
def obtener_topicos_lsa(
    modelo: TruncatedSVD,
    vocabulario: np.ndarray,
    top_n: int = TOP_N,
) -> tuple[list[list[str]], list[list[float]]]:
    """Extrae términos representativos de cada componente LSA.
    Para identificar las características de mayor peso se utiliza la magnitud
    absoluta del coeficiente. El peso firmado se conserva para exportarlo.
    """
    topicos: list[list[str]] = []
    pesos_topicos: list[list[float]] = []

    for componente in modelo.components_:
        indices = np.argsort(
            np.abs(componente)
        )[::-1][:top_n]

        palabras = [
            str(vocabulario[indice])
            for indice in indices
        ]

        pesos = [
            float(componente[indice])
            for indice in indices
        ]

        topicos.append(palabras)
        pesos_topicos.append(pesos)

    return topicos, pesos_topicos


def calcular_coherencia_topicos(
    topicos: list[list[str]],
    documentos_tokens: list[list[str]],
    diccionario: Dictionary,
) -> float:
    """Calcula Coherence Score c_v para una lista de tópicos."""
    modelo_coherencia = CoherenceModel(
        topics=topicos,
        texts=documentos_tokens,
        dictionary=diccionario,
        coherence="c_v",
        processes=1,
    )

    return float(
        modelo_coherencia.get_coherence()
    )


def evaluar_lsa(
    matriz_tfidf,
    vocabulario: np.ndarray,
    documentos_tokens: list[list[str]],
    diccionario: Dictionary,
):
    """Evalúa LSA para todos los valores de k."""
    # L2 por documento: sin esto, 3-4 documentos muy largos dominan el SVD
    matriz_lsa = normalize(matriz_tfidf, norm="l2")
    resultados: list[dict[str, float | int]] = []

    mejor_modelo = None
    mejores_topicos = None
    mejores_pesos = None
    mejor_k = None
    mejor_coherencia = -np.inf

    for k in VALORES_K:
        print(f"  LSA | k={k}")

        modelo = TruncatedSVD(
            n_components=k,
            random_state=RANDOM_STATE,
        )

        modelo.fit(matriz_lsa)

        topicos, pesos = obtener_topicos_lsa(
            modelo,
            vocabulario,
            top_n=TOP_N,
        )

        coherencia = calcular_coherencia_topicos(
            topicos,
            documentos_tokens,
            diccionario,
        )

        resultados.append(
            {
                "k": k,
                "coherencia_lsa": coherencia,
                "varianza_explicada_acumulada": float(
                    modelo.explained_variance_ratio_.sum()
                ),
            }
        )

        if coherencia > mejor_coherencia:
            mejor_coherencia = coherencia
            mejor_k = k
            mejor_modelo = modelo
            mejores_topicos = topicos
            mejores_pesos = pesos

    return (
        pd.DataFrame(resultados),
        mejor_modelo,
        mejores_topicos,
        mejores_pesos,
        mejor_k,
        float(mejor_coherencia),
    )



# LDA
def obtener_topicos_lda(
    modelo: LdaModel,
    top_n: int = TOP_N,
) -> tuple[list[list[str]], list[list[float]]]:
    """Extrae las palabras y probabilidades más representativas por tema."""
    topicos: list[list[str]] = []
    probabilidades: list[list[float]] = []

    for indice_tema in range(modelo.num_topics):
        terminos = modelo.show_topic(
            indice_tema,
            topn=top_n,
        )

        topicos.append([
            palabra
            for palabra, _ in terminos
        ])

        probabilidades.append([
            float(probabilidad)
            for _, probabilidad in terminos
        ])

    return topicos, probabilidades


def evaluar_lda(
    corpus_bow,
    diccionario: Dictionary,
    documentos_tokens: list[list[str]],
):
    """Evalúa LDA para todos los valores de k."""
    resultados: list[dict[str, float | int]] = []

    mejor_modelo = None
    mejores_topicos = None
    mejores_probabilidades = None
    mejor_k = None
    mejor_coherencia = -np.inf

    for k in VALORES_K:
        print(f"  LDA | k={k}")

        modelo = LdaModel(
            corpus=corpus_bow,
            id2word=diccionario,
            num_topics=k,
            random_state=RANDOM_STATE,
            passes=LDA_PASSES,
            iterations=LDA_ITERATIONS,
            eval_every=None,
        )

        topicos, probabilidades = obtener_topicos_lda(
            modelo,
            top_n=TOP_N,
        )

        coherencia = CoherenceModel(
            model=modelo,
            texts=documentos_tokens,
            dictionary=diccionario,
            coherence="c_v",
            processes=1,
        ).get_coherence()

        coherencia = float(coherencia)

        resultados.append(
            {
                "k": k,
                "coherencia_lda": coherencia,
            }
        )

        if coherencia > mejor_coherencia:
            mejor_coherencia = coherencia
            mejor_k = k
            mejor_modelo = modelo
            mejores_topicos = topicos
            mejores_probabilidades = probabilidades

    return (
        pd.DataFrame(resultados),
        mejor_modelo,
        mejores_topicos,
        mejores_probabilidades,
        mejor_k,
        float(mejor_coherencia),
    )



# Exportación de resultados
def guardar_topicos_lsa(
    topicos: list[list[str]],
    pesos: list[list[float]],
    ruta: Path,
) -> None:
    """Guarda los temas LSA e interpretación manual."""
    registros = []

    for indice_tema, (palabras, pesos_tema) in enumerate(
        zip(topicos, pesos),
        start=1,
    ):
        registros.append(
            {
                "tema": f"Tema {indice_tema}",
                "palabras_representativas": " | ".join(palabras),
                "pesos": " | ".join(
                    f"{peso:.6f}"
                    for peso in pesos_tema
                ),
                
                "interpretacion_manual": INTERPRETACIONES_LSA.get(
                    f"Tema {indice_tema}", ""
                ),
            }
        )

    pd.DataFrame(registros).to_csv(
        ruta,
        index=False,
        encoding="utf-8",
    )


def guardar_topicos_lda(
    topicos: list[list[str]],
    probabilidades: list[list[float]],
    ruta: Path,
) -> None:
    """Guarda los temas LDA y sus términos más representativos."""
    registros = []

    for indice_tema, (palabras, probabilidades_tema) in enumerate(
        zip(topicos, probabilidades),
        start=1,
    ):
        registros.append(
            {
                "tema": f"Tema {indice_tema}",
                "palabras_representativas": " | ".join(palabras),
                "probabilidades": " | ".join(
                    f"{probabilidad:.6f}"
                    for probabilidad in probabilidades_tema
                ),
                "interpretacion": INTERPRETACIONES_LDA.get(
                    f"Tema {indice_tema}", ""
                ),
            }
        )

    pd.DataFrame(registros).to_csv(
        ruta,
        index=False,
        encoding="utf-8",
    )


def guardar_grafica_coherencia(
    resultados: pd.DataFrame,
    ruta: Path,
) -> None:
    """Genera k vs. Score de Coherencia para LSA y LDA."""
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        resultados["k"],
        resultados["coherencia_lsa"],
        marker="o",
        label="LSA",
    )

    ax.plot(
        resultados["k"],
        resultados["coherencia_lda"],
        marker="o",
        label="LDA",
    )

    ax.set_xlabel("Número de temas (k)")
    ax.set_ylabel("Score de Coherencia (c_v)")
    ax.set_title("Número de temas vs. Score de Coherencia")
    ax.set_xticks(VALORES_K)
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        ruta,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


# Flujo principal
def modelar_temas() -> None:
    """Ejecuta exclusivamente el punto 7 de la práctica."""
    print("Punto 7 - Modelado de temas")
    print("-------------:)---------------")

    (
        df,
        matriz_tfidf,
        vocabulario,
    ) = cargar_entradas()

    documentos_tokens = [
        deserializar_caracteristicas(valor)
        for valor in df[COLUMNA_BOW]
    ]

    if not any(documentos_tokens):
        raise RuntimeError(
            f"La columna {COLUMNA_BOW!r} no contiene características."
        )

    print(f"Documentos: {len(documentos_tokens)}")
    print(f"Características TF-IDF: {matriz_tfidf.shape[1]}")
    print()

    diccionario, corpus_bow = preparar_corpus_gensim(
        documentos_tokens
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)


    # 7.1 + 7.3: LSA y coherencia para k = 2...10
    print("Evaluando LSA...")

    (
        resultados_lsa,
        mejor_lsa,
        topicos_lsa,
        pesos_lsa,
        mejor_k_lsa,
        coherencia_lsa,
    ) = evaluar_lsa(
        matriz_tfidf,
        vocabulario,
        documentos_tokens,
        diccionario,
    )

    
    # 7.2 + 7.3: LDA y coherencia para k = 2...10
    print()
    print("Evaluando LDA...")

    (
        resultados_lda,
        mejor_lda,
        topicos_lda,
        probabilidades_lda,
        mejor_k_lda,
        coherencia_lda,
    ) = evaluar_lda(
        corpus_bow,
        diccionario,
        documentos_tokens,
    )

    # Resultados de coherencia
    resultados_coherencia = resultados_lsa.merge(
        resultados_lda,
        on="k",
        how="inner",
    )

    ruta_coherencia = OUTPUT_DIR / "coherencia_modelos.csv"

    resultados_coherencia.to_csv(
        ruta_coherencia,
        index=False,
        encoding="utf-8",
    )

    ruta_grafica = OUTPUT_DIR / "grafica_coherencia.png"

    guardar_grafica_coherencia(
        resultados_coherencia,
        ruta_grafica,
    )


    # Temas finales 
    ruta_lsa = OUTPUT_DIR / "temas_lsa.csv"
    guardar_topicos_lsa(
        topicos_lsa,
        pesos_lsa,
        ruta_lsa,
    )

    ruta_lda = OUTPUT_DIR / "temas_lda.csv"
    guardar_topicos_lda(
        topicos_lda,
        probabilidades_lda,
        ruta_lda,
    )

    # Guardar modelos para reutilizarlos desoues.
    ruta_modelo_lsa = MODELOS_DIR / "modelo_lsa.joblib"
    joblib.dump(
        mejor_lsa,
        ruta_modelo_lsa,
    )

    ruta_modelo_lda = MODELOS_DIR / "modelo_lda.gensim"
    mejor_lda.save(
        str(ruta_modelo_lda)
    )

    ruta_diccionario = MODELOS_DIR / "diccionario_gensim.dict"
    diccionario.save(
        str(ruta_diccionario)
    )

    # Resumen / selección según coherencia
    resumen = {
        "documentos": int(len(df)),
        "representacion_lsa": "TF-IDF del punto 6",
        "representacion_lda": (
            f"Bag of Words de la columna {COLUMNA_BOW} del punto 5"
        ),
        "rango_k_evaluado": VALORES_K,
        "score_coherencia": "c_v",
        "top_palabras_por_tema": TOP_N,
        "lsa": {
            "metodo": "Truncated SVD",
            "k_seleccionado": int(mejor_k_lsa),
            "coherencia": coherencia_lsa,
            "criterio_seleccion": (
                "Mayor Score de Coherencia c_v dentro del rango evaluado"
            ),
            "requiere_interpretacion_manual": True,
        },
        "lda": {
            "metodo": "Gensim LdaModel",
            "k_seleccionado": int(mejor_k_lda),
            "coherencia": coherencia_lda,
            "criterio_seleccion": (
                "Mayor Score de Coherencia c_v dentro del rango evaluado"
            ),
            "passes": LDA_PASSES,
            "iterations": LDA_ITERATIONS,
        },
        "archivos_salida": {
            "coherencia": str(ruta_coherencia),
            "grafica_coherencia": str(ruta_grafica),
            "temas_lsa": str(ruta_lsa),
            "temas_lda": str(ruta_lda),
            "modelo_lsa": str(ruta_modelo_lsa),
            "modelo_lda": str(ruta_modelo_lda),
            "diccionario_gensim": str(ruta_diccionario),
        },
        "nota": (
            "El punto 8 no se ejecuta aquí. La asignación del tema "
            "dominante por documento y la comparación Verdadero/Falso "
            "se realizarán posteriormente."
        ),
    }

    ruta_resumen = OUTPUT_DIR / "resumen_modelado_temas.json"

    ruta_resumen.write_text(
        json.dumps(
            resumen,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Modelado completado")
    print("-------------------")
    print(
        f"LSA -> mejor k: {mejor_k_lsa} | "
        f"coherencia: {coherencia_lsa:.4f}"
    )
    print(
        f"LDA -> mejor k: {mejor_k_lda} | "
        f"coherencia: {coherencia_lda:.4f}"
    )
    print()
    print(f"Resultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    modelar_temas()

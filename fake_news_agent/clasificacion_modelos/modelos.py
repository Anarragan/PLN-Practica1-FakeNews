"""Experimentos preliminares de clasificacion para el punto 10 de la guia."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
AGENT_DIR = BASE_DIR.parent
CORPUS_PATH = AGENT_DIR / "preprocesamiento" / "corpus_normalizado.csv"
OUTPUT_DIR = BASE_DIR
TOKEN_PATTERN = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", flags=re.UNICODE)
RANDOM_STATE = 42
NUM_FOLDS = 10
TEST_SIZE = 0.2
MIN_FEATURE_FREQUENCY = 3


@dataclass(frozen=True)
class Documento:
	nombre: str
	texto: str
	etiqueta: int


class AnalizadorTexto:
	"""Tokeniza y aplica las transformaciones configuradas antes de BoW."""

	def __init__(self, usar_stopwords: bool, usar_stemming: bool) -> None:
		self.usar_stopwords = usar_stopwords
		self.usar_stemming = usar_stemming
		self.stopwords_es = cargar_stopwords_espanol() if usar_stopwords else set()
		self.stemmer = SnowballStemmer("spanish") if usar_stemming else None

	def __call__(self, texto: str) -> list[str]:
		tokens = [token.lower() for token in TOKEN_PATTERN.findall(texto)]
		caracteristicas = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
		if self.usar_stopwords:
			caracteristicas = [
				caracteristica
				for caracteristica in caracteristicas
				if not any(token in self.stopwords_es for token in caracteristica.split())
			]
		if self.stemmer is not None:
			caracteristicas = [
				" ".join(self.stemmer.stem(token) for token in caracteristica.split())
				for caracteristica in caracteristicas
			]
		return caracteristicas


def cargar_stopwords_espanol() -> set[str]:
	"""Carga las stopwords NLTK en español y descarga el recurso si falta."""
	try:
		return set(stopwords.words("spanish"))
	except LookupError:
		nltk.download("stopwords", quiet=True)
		return set(stopwords.words("spanish"))


def cargar_corpus(ruta: Path = CORPUS_PATH) -> list[Documento]:
	"""Carga los textos normalizados y las etiquetas del punto 2."""
	if not ruta.is_file():
		raise FileNotFoundError(
			f"No existe el corpus normalizado: {ruta}\n"
			"Ejecuta primero fake_news_agent/preprocesamiento/normalizer.py."
		)
	columnas_requeridas = {"archivo", "label", "texto_normalizado"}
	documentos: list[Documento] = []
	with ruta.open(newline="", encoding="utf-8") as archivo:
		lector = csv.DictReader(archivo)
		if lector.fieldnames is None or not columnas_requeridas.issubset(lector.fieldnames):
			faltantes = columnas_requeridas - set(lector.fieldnames or [])
			raise ValueError(
				"El corpus normalizado no contiene las columnas requeridas: "
				f"{sorted(faltantes)}"
			)
		for fila in lector:
			try:
				etiqueta = int(fila["label"])
			except (TypeError, ValueError) as error:
				raise ValueError(f"Etiqueta inválida en {fila.get('archivo', '')!r}.") from error
			if etiqueta not in (0, 1):
				raise ValueError(f"Etiqueta {etiqueta} inválida; se esperaba 0 o 1.")
			documentos.append(
				Documento(
					nombre=fila["archivo"],
					texto=fila["texto_normalizado"] or "",
					etiqueta=etiqueta,
				)
			)
	if not documentos:
		raise ValueError("El corpus normalizado no contiene documentos.")
	if {documento.etiqueta for documento in documentos} != {0, 1}:
		raise ValueError("El corpus debe contener documentos de ambas clases (0 y 1).")
	return documentos


def generar_configuraciones() -> list[dict[str, object]]:
	"""Genera las ocho combinaciones de ponderacion y reduccion del punto 10."""
	return [
		{
			"ponderacion": ponderacion,
			"stopwords": usar_stopwords,
			"stemming": usar_stemming,
			"reduccion": (
				"ambas"
				if usar_stopwords and usar_stemming
				else "solo_stopwords"
				if usar_stopwords
				else "solo_stemming"
				if usar_stemming
				else "ninguna"
			),
		}
		for ponderacion in ("TO", "TF-IDF")
		for usar_stopwords, usar_stemming in (
			(False, False),
			(True, False),
			(False, True),
			(True, True),
		)
	]


class FiltroFrecuencia(BaseEstimator, TransformerMixin):
	"""Retiene caracteristicas con al menos N ocurrencias en el train del fold."""

	def __init__(self, frecuencia_minima: int = MIN_FEATURE_FREQUENCY) -> None:
		self.frecuencia_minima = frecuencia_minima

	def fit(self, X: object, y: object = None) -> FiltroFrecuencia:
		frecuencias = np.asarray(X.sum(axis=0)).ravel()
		self.caracteristicas_validas_ = frecuencias >= self.frecuencia_minima
		if not np.any(self.caracteristicas_validas_):
			raise ValueError(
				"No quedan caracteristicas con la frecuencia minima configurada."
			)
		return self

	def transform(self, X: object) -> object:
		return X[:, self.caracteristicas_validas_]


def crear_vectorizador(configuracion: dict[str, object]) -> CountVectorizer:
	"""Construye BoW con unigramas/bigramas para ajuste dentro de cada fold."""
	analizador = AnalizadorTexto(
		usar_stopwords=bool(configuracion["stopwords"]),
		usar_stemming=bool(configuracion["stemming"]),
	)
	return CountVectorizer(analyzer=analizador, dtype=np.int32)


def crear_modelos() -> dict[str, BaseEstimator]:
	"""Retorna los cuatro clasificadores solicitados, con semilla fija."""
	return {
		"Regresion logistica": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
		"Arbol de decision": DecisionTreeClassifier(random_state=RANDOM_STATE),
		"KNN": KNeighborsClassifier(n_neighbors=5),
		"SVM": SVC(kernel="linear", random_state=RANDOM_STATE),
	}


def calcular_metricas(y_real: Iterable[int], y_predicho: Iterable[int]) -> dict[str, float]:
	"""Calcula las metricas requeridas usando promediado ponderado por clase."""
	reales = list(y_real)
	predichos = list(y_predicho)
	return {
		"accuracy": float(accuracy_score(reales, predichos)),
		"precision_weighted": float(
			precision_score(reales, predichos, average="weighted", zero_division=0)
		),
		"recall_weighted": float(
			recall_score(reales, predichos, average="weighted", zero_division=0)
		),
		"f1_weighted": float(
			f1_score(reales, predichos, average="weighted", zero_division=0)
		),
	}


def evaluar_modelo(
	textos_entrenamiento: list[str],
	etiquetas_entrenamiento: list[int],
	textos_prueba: list[str],
	etiquetas_prueba: list[int],
	configuracion: dict[str, object],
	nombre_modelo: str,
	modelo: BaseEstimator,
	frecuencia_minima: int = MIN_FEATURE_FREQUENCY,
	folds: int = NUM_FOLDS,
) -> dict[str, object]:
	"""Evalua una configuracion con CV estratificada y un test independiente."""
	pipeline = Pipeline(
		[
			("vectorizador", crear_vectorizador(configuracion)),
			("filtro_frecuencia", FiltroFrecuencia(frecuencia_minima)),
		]
	)
	if configuracion["ponderacion"] == "TF-IDF":
		pipeline.steps.append(
			("ponderador", TfidfTransformer(norm=None, use_idf=True, smooth_idf=True))
		)
	pipeline.steps.append(("clasificador", modelo))
	cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
	puntuaciones = cross_validate(
		pipeline,
		textos_entrenamiento,
		etiquetas_entrenamiento,
		cv=cv,
		scoring=("accuracy", "precision_weighted", "recall_weighted", "f1_weighted"),
		n_jobs=1,
		error_score="raise",
	)

	pipeline.fit(textos_entrenamiento, etiquetas_entrenamiento)
	predicciones = pipeline.predict(textos_prueba)
	resultado: dict[str, object] = {
		"modelo": nombre_modelo,
		**configuracion,
		"frecuencia_minima": frecuencia_minima,
		"folds": folds,
	}
	for metrica in ("accuracy", "precision_weighted", "recall_weighted", "f1_weighted"):
		valores = puntuaciones[f"test_{metrica}"]
		resultado[f"cv_{metrica}_mean"] = float(np.mean(valores))
		resultado[f"cv_{metrica}_std"] = float(np.std(valores))
	resultado.update(
		{
			f"test_{metrica}": valor
			for metrica, valor in calcular_metricas(etiquetas_prueba, predicciones).items()
		}
	)
	return resultado


def ejecutar_experimentos(
	documentos: list[Documento] | None = None,
	directorio_salida: Path = OUTPUT_DIR,
	frecuencia_minima: int = MIN_FEATURE_FREQUENCY,
	folds: int = NUM_FOLDS,
) -> list[dict[str, object]]:
	"""Ejecuta las 32 evaluaciones (8 configuraciones por 4 algoritmos)."""
	documentos = documentos if documentos is not None else cargar_corpus()
	textos = [documento.texto for documento in documentos]
	etiquetas = [documento.etiqueta for documento in documentos]
	indices = np.arange(len(documentos))
	indices_train, indices_test = train_test_split(
		indices,
		test_size=TEST_SIZE,
		random_state=RANDOM_STATE,
		stratify=etiquetas,
	)
	textos_train = [textos[indice] for indice in indices_train]
	textos_test = [textos[indice] for indice in indices_test]
	etiquetas_train = [etiquetas[indice] for indice in indices_train]
	etiquetas_test = [etiquetas[indice] for indice in indices_test]

	resultados: list[dict[str, object]] = []
	for configuracion in generar_configuraciones():
		for nombre_modelo, modelo in crear_modelos().items():
			resultado = evaluar_modelo(
				textos_train,
				etiquetas_train,
				textos_test,
				etiquetas_test,
				configuracion,
				nombre_modelo,
				modelo,
				frecuencia_minima=frecuencia_minima,
				folds=folds,
			)
			resultados.append(resultado)
			print(
				f"{nombre_modelo} | {configuracion['ponderacion']} | "
				f"{configuracion['reduccion']}: "
				f"F1 CV={resultado['cv_f1_weighted_mean']:.4f}, "
				f"F1 test={resultado['test_f1_weighted']:.4f}"
			)

	directorio_salida.mkdir(parents=True, exist_ok=True)
	guardar_resultados(resultados, directorio_salida / "metricas_modelos.csv")
	guardar_graficos(resultados, directorio_salida)
	conteo_clases = {
		"Verdad (1)": sum(documento.etiqueta == 1 for documento in documentos),
		"Falso (0)": sum(documento.etiqueta == 0 for documento in documentos),
	}
	guardar_analisis(
		resultados,
		directorio_salida / "analisis_resultados.md",
		conteo_clases,
	)
	(directorio_salida / "configuracion_experimentos.json").write_text(
		json.dumps(
			{
				"documentos": len(documentos),
				"distribucion_clases": conteo_clases,
				"entrenamiento": len(indices_train),
				"prueba": len(indices_test),
				"etiquetas": {"Verdad": 1, "Falso": 0},
				"semilla": RANDOM_STATE,
				"folds": folds,
				"frecuencia_minima": frecuencia_minima,
				"referencia": "F1 ponderado",
			},
			ensure_ascii=False,
			indent=2,
		),
		encoding="utf-8",
	)
	return resultados


def guardar_resultados(resultados: list[dict[str, object]], ruta: Path) -> None:
	"""Exporta métricas CV y test a un CSV listo para tablas/informe."""
	if not resultados:
		raise ValueError("No hay resultados para guardar.")
	with ruta.open("w", newline="", encoding="utf-8") as archivo:
		escritor = csv.DictWriter(archivo, fieldnames=list(resultados[0]))
		escritor.writeheader()
		escritor.writerows(resultados)


def guardar_graficos(resultados: list[dict[str, object]], directorio: Path) -> None:
	"""Genera las tres graficas solicitadas en la guia, sobre F1 de test."""
	import matplotlib.pyplot as plt

	modelos = list(crear_modelos())
	maximos = {
		modelo: max(
			float(fila["cv_f1_weighted_mean"])
			for fila in resultados
			if fila["modelo"] == modelo
		)
		for modelo in modelos
	}
	fig, ax = plt.subplots(figsize=(9, 5))
	ax.bar(maximos.keys(), maximos.values(), color="#277da1")
	ax.set(
		title="F1-score maximo por algoritmo",
		ylabel="F1 ponderado medio (CV)",
		ylim=(0, 1),
	)
	ax.tick_params(axis="x", rotation=15)
	fig.tight_layout()
	fig.savefig(directorio / "f1_maximo_por_algoritmo.png", dpi=160)
	plt.close(fig)

	ponderaciones = ("TO", "TF-IDF")
	valores_ponderacion = {
		(ponderacion, modelo): _media_f1(
			fila
			for fila in resultados
			if fila["ponderacion"] == ponderacion and fila["modelo"] == modelo
		)
		for ponderacion in ponderaciones
		for modelo in modelos
	}
	_guardar_grafico_grupos(
		valores_ponderacion,
		list(ponderaciones),
		modelos,
		"F1-score medio por ponderacion y algoritmo",
		directorio / "f1_medio_ponderacion_algoritmo.png",
	)

	reducciones = ("ninguna", "solo_stopwords", "solo_stemming", "ambas")
	valores_reduccion = {
		(reduccion, modelo): _media_f1(
			fila
			for fila in resultados
			if fila["reduccion"] == reduccion and fila["modelo"] == modelo
		)
		for reduccion in reducciones
		for modelo in modelos
	}
	_guardar_grafico_grupos(
		valores_reduccion,
		list(reducciones),
		modelos,
		"F1-score medio por reduccion y algoritmo",
		directorio / "f1_medio_reduccion_algoritmo.png",
	)


def _media_f1(filas: Iterable[dict[str, object]]) -> float:
	valores = [float(fila["cv_f1_weighted_mean"]) for fila in filas]
	return float(np.mean(valores)) if valores else 0.0


def _guardar_grafico_grupos(
	valores: dict[tuple[str, str], float],
	grupos: list[str],
	modelos: list[str],
	titulo: str,
	ruta: Path,
) -> None:
	import matplotlib.pyplot as plt

	posiciones = np.arange(len(grupos))
	ancho = 0.8 / len(modelos)
	fig, ax = plt.subplots(figsize=(10, 5))
	for indice, modelo in enumerate(modelos):
		desplazamiento = (indice - (len(modelos) - 1) / 2) * ancho
		ax.bar(
			posiciones + desplazamiento,
			[valores[(grupo, modelo)] for grupo in grupos],
			ancho,
			label=modelo,
		)
	ax.set(
		title=titulo,
		ylabel="F1 ponderado medio (CV)",
		ylim=(0, 1),
		xticks=posiciones,
		xticklabels=grupos,
	)
	ax.legend()
	fig.tight_layout()
	fig.savefig(ruta, dpi=160)
	plt.close(fig)


def guardar_analisis(
	resultados: list[dict[str, object]],
	ruta: Path,
	conteo_clases: dict[str, int],
) -> None:
	"""Genera tablas completas y un resumen visual de los resultados."""
	mejor_global = max(resultados, key=lambda fila: float(fila["cv_f1_weighted_mean"]))
	lineas = [
		"# Punto 10 | Clasificación de noticias",
		"",
		"> Comparación de cuatro clasificadores, dos esquemas de ponderación y cuatro "
		"variantes de preprocesamiento. La métrica de referencia es el F1 ponderado.",
		"",
		"## Diseño experimental",
		"",
		f"**Corpus:** {sum(conteo_clases.values())} documentos, "
		f"{conteo_clases['Verdad (1)']} verdaderos y {conteo_clases['Falso (0)']} falsos. "
		"**Evaluación:** partición estratificada 80/20 y validación cruzada estratificada "
		"de 10 folds sobre entrenamiento. Las métricas de CV se presentan como media; "
		"para F1 también se muestra la desviación estándar.",
		"",
		"## Vista general",
		"",
		"![F1 máximo por algoritmo](f1_maximo_por_algoritmo.png)",
		"",
		"![F1 medio por ponderación y algoritmo](f1_medio_ponderacion_algoritmo.png)",
		"",
		"![F1 medio por reducción y algoritmo](f1_medio_reduccion_algoritmo.png)",
		"",
		"## Resultados por algoritmo",
		"",
		"Accuracy es la proporción de aciertos; precisión, recall y F1 se calculan "
		"con promedio ponderado por clase. Cada tabla contiene las ocho configuraciones "
		"en el mismo orden de la guía.",
	]
	columnas_metricas = (
		("Accuracy", "accuracy"),
		("Precisión ponderada", "precision_weighted"),
		("Recall ponderado", "recall_weighted"),
		("F1 ponderado", "f1_weighted"),
	)
	for modelo in crear_modelos():
		filas_modelo = [fila for fila in resultados if fila["modelo"] == modelo]
		lineas.extend(["", f"### {modelo}", "", "**Validación cruzada (10 folds)**", ""])
		lineas.append(
			"| # | Ponderación | Stopwords | Stemming | "
			+ " | ".join(nombre for nombre, _ in columnas_metricas)
			+ " |"
		)
		lineas.append("|---:|---|:---:|:---:|" + "---:|" * len(columnas_metricas))
		for numero, fila in enumerate(filas_modelo, start=1):
			valores = []
			for _, metrica in columnas_metricas:
				media = float(fila[f"cv_{metrica}_mean"])
				if metrica == "f1_weighted":
					desviacion = float(fila["cv_f1_weighted_std"])
					valores.append(f"{media * 100:.2f}% ± {desviacion * 100:.2f}%")
				else:
					valores.append(f"{media * 100:.2f}%")
			lineas.append(
				f"| {numero} | {fila['ponderacion']} | "
				f"{'Sí' if fila['stopwords'] else 'No'} | "
				f"{'Sí' if fila['stemming'] else 'No'} | "
				+ " | ".join(valores)
				+ " |"
			)
		lineas.extend(["", "**Conjunto de prueba (20%)**", ""])
		lineas.append(
			"| # | Ponderación | Stopwords | Stemming | "
			+ " | ".join(nombre for nombre, _ in columnas_metricas)
			+ " |"
		)
		lineas.append("|---:|---|:---:|:---:|" + "---:|" * len(columnas_metricas))
		for numero, fila in enumerate(filas_modelo, start=1):
			valores = [
				f"{float(fila[f'test_{metrica}']) * 100:.2f}%"
				for _, metrica in columnas_metricas
			]
			lineas.append(
				f"| {numero} | {fila['ponderacion']} | "
				f"{'Sí' if fila['stopwords'] else 'No'} | "
				f"{'Sí' if fila['stemming'] else 'No'} | "
				+ " | ".join(valores)
				+ " |"
			)
	lineas.extend(
		[
			"",
			"## Comparación y observaciones",
			"",
			f"La mejor configuración según F1 ponderado en validación cruzada fue "
			f"**{mejor_global['modelo']}** con ponderación **{mejor_global['ponderacion']}**, "
			f"stopwords={mejor_global['stopwords']} y stemming={mejor_global['stemming']} "
			f"(F1 CV={float(mejor_global['cv_f1_weighted_mean']) * 100:.2f}%; "
			f"F1 test={float(mejor_global['test_f1_weighted']) * 100:.2f}%).",
			"",
			"Las gráficas resumen el máximo de F1 por algoritmo y las medias al cambiar "
			"la ponderación o las técnicas de reducción. Las tablas anteriores contienen "
			"las cuatro métricas para cada combinación, tanto en CV como en prueba.",
			"",
			"Los resultados deben interpretarse considerando la composición del corpus: "
			"incluye ejemplos de la segunda fuente, que el proyecto describe como sintéticos. "
			"Por ello, un buen resultado de clasificación no garantiza por sí solo la misma "
			"capacidad de generalización ante noticias reales de otras fuentes.",
			"",
			"## Archivos generados",
			"",
			"- `metricas_modelos.csv`: métricas de validación cruzada y prueba para las 32 configuraciones.",
			"- `f1_maximo_por_algoritmo.png`, `f1_medio_ponderacion_algoritmo.png` y `f1_medio_reduccion_algoritmo.png`: gráficas comparativas incluidas arriba.",
			"- `configuracion_experimentos.json`: tamaños, etiquetas y parámetros de la ejecución.",
		]
	)
	ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def main() -> None:
	"""Ejecuta la comparación completa del punto 10."""
	ejecutar_experimentos()


if __name__ == "__main__":
	main()


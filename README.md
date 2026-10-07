# PLN-Práctica 1: Fake News

### Guía rápida de trabajo
Para trabajar de forma ordenada sin sobrescribir el trabajo de los demás, seguiremos este flujo de ramas:

1. **`main`**: Versión estable y final.
2. **`develop`**: Rama base para integrar todos los avances.
3. **Tu rama personal (`tu-nombre`)**: Rama propia donde trabajarás en tus tareas asignadas.

#### Flujo de trabajo en Git
* **Inicio:** Al clonar el repositorio estarás en `main`. Debes cambiar a `develop` y crear tu rama personal a partir de ella.
* **Desarrollo:** Realiza tus cambios y confirmaciones (*commits*) exclusivamente en tu rama personal.
* **Integración:** Para compartir tus avances, debes hacer un *Pull Request* (PR) hacia la rama `develop`. Una vez que `develop` se actualice con los cambios de todos, los demás miembros deben actualizar sus respectivas ramas personales desde `develop`.
* **Cierre de tareas:** Al terminar un ítem, se revisan los cambios de forma conjunta y se hace un *Pull Request* final de `develop` hacia `main` para asegurar el control de versiones ante fallos posteriores.
* **Herramientas:** Se permite el uso de GitHub Desktop para facilitar la gestión del flujo.

> ⚠️ **Aclaración importante para los PR:** En GitHub, al hacer *commit* en tu rama personal, aparecerá un recuadro naranja con el texto *"Compare & pull request"*. Al hacer clic, asegúrate de configurar correctamente los destinos: **`base: develop`** y **`compare: tu-nombre`**.

---

### Guía de ejecución
#### Preparar el entorno

Se recomienda Python 3.12. Desde la raíz del repositorio, crea un entorno virtual e instala las dependencias declaradas por el proyecto:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export NLTK_DATA="$PWD/.venv/nltk_data"
export KAGGLEHUB_CACHE="$PWD/.cache/kagglehub"
```

En Windows, crea el entorno con `py -3.12 -m venv .venv` y actívalo con `.venv\\Scripts\\Activate.ps1` en PowerShell. Después de activarlo, ejecuta `python -m pip install -r requirements.txt` y configura las cachés locales:

```powershell
$env:NLTK_DATA = "$PWD\\.venv\\nltk_data"
$env:KAGGLEHUB_CACHE = "$PWD\\.cache\\kagglehub"
```

La instalación queda dentro del entorno, no en la instalación global de Python. `.venv` y `.cache` se mantienen fuera de Git; para salir del entorno ejecuta `deactivate`.

* **Nota de orden obligatoria:** Es necesario ejecutar primero el archivo `scraper.py` para generar la estructura básica de almacenamiento de datos antes de procesar la segunda fuente.

Con el entorno activado, puede ejecutar los scripts desde la raíz del repositorio:

```bash
python fake_news_agent/corpus/scraper.py
python fake_news_agent/corpus/fuente_2.py
python fake_news_agent/corpus/metricas.py
```

---

### Notas y mejoras futuras
* **Fuente 2:** El corpus actual de 5000 muestras es sintético. Se debería sustituir o complementar con un conjunto de datos (*dataset*) verificado por humanos.

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
* **Nota de orden obligatoria:** Es necesario ejecutar primero el archivo `scraper.py` para generar la estructura básica de almacenamiento de datos antes de procesar la segunda fuente.

---

### Estado del proyecto

#### Completado
- [x] Obtener el conjunto de 50 archivos `.txt` mediante *web scraping* (si no se ha completado, cambiar `N_DOCS_PER_CLASS = 75`).
- [x] Almacenar los archivos en dos carpetas independientes: verdadero y falso.
- [x] Conseguir las 5000 muestras de la fuente 2 (de forma aleatoria y balanceada).
- [x] Calcular las métricas base.

#### En proceso
- [ ] 2. Preprocesamiento de texto (limpieza, eliminación de *stop words*, tokenización, etc.).

---

### Notas y mejoras futuras
* **Fuente 2:** El corpus actual de 5000 muestras es sintético. Se debería sustituir o complementar con un conjunto de datos (*dataset*) verificado por humanos.

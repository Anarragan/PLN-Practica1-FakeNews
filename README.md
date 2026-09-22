# PLN-Practica1-FakeNews

### Guia rapida de trabajo
Para trabajar de forma ordenada sin sobrescribir el trabajo de los demás, seguiremos este flujo:

1. **`main`**: Versión estable y final.
2. **`develop`**: Rama base para integrar todos los avances.
3. **Tu rama personal (`tu-nombre`)**: Donde trabajarás en tus tareas.

- Al clonar, estarás en `main`. Debes pasarte a `develop` y crear tu rama desde allí.
- Trabaja en la rama con tu nombre (commits).
- Para compartir los cambios se debe hacer pull request a la rama develop y luego actualizar la rama del desarrollador (develop cambia entonces los demas debemos actualizar).

rama_nombre -> develop
develop -> rama_nombre

- Al terminar un item se revisan los cambios y se hace pull request a main para tener control en caso de fallos posteriores.
- Se puede usar github desktop para mayor facilidad.

**ACLARACION PARA PR**: En github al hacer commit en tu rama va a aparecer un cuadrado naranja con el texto "compare and pull request" al hacer click debes fijarte que base:develop y compare:rama_tu_nombre

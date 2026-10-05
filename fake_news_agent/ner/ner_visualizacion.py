import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "ner_entidades.csv")


# --- 1. Total de entidades por clase ---
def total_por_clase(df):
    return df["Clase"].value_counts()


# --- 2. Distribución porcentual por tipo de entidad ---
def distribucion_por_tipo(df):
    tabla = df.groupby(["Clase", "Tipo"]).size().unstack(fill_value=0)
    porcentajes = tabla.div(tabla.sum(axis=1), axis=0) * 100
    return tabla, porcentajes


# --- 3. Top 10 entidades por tipo, para cada clase ---
def top_10_por_tipo(df, clase, tipo, n=10):
    subset = df[(df["Clase"] == clase) & (df["Tipo"] == tipo)]
    return subset["Entidad"].str.lower().value_counts().head(n)


# --- 4. Top 20 entidades globales por clase (sin importar el tipo) ---
def top_20_global(df, clase, n=20):
    subset = df[df["Clase"] == clase]
    return subset["Entidad"].str.lower().value_counts().head(n)


# --- Gráfico A: distribución comparativa de tipos de entidad ---
def graficar_distribucion_tipos(porcentajes):
    porcentajes.T.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("Porcentaje de entidades (%)")
    plt.title("Distribución de tipos de entidad: Verdadero vs Falso")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "ner_distribucion_tipos.png", dpi=150)
    plt.show()


# --- Gráfico B/C: entidades más frecuentes por clase ---
def graficar_top_entidades(df, clase, n=15):
    top = top_20_global(df, clase, n=n)
    top.plot(kind="barh", figsize=(8, 6))
    plt.gca().invert_yaxis()
    plt.xlabel("Frecuencia")
    plt.title(f"Entidades más frecuentes - {clase.upper()}")
    plt.tight_layout()
    plt.savefig(BASE_DIR / f"ner_top_entidades_{clase}.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    print("Total de entidades por clase:")
    print(total_por_clase(df), "\n")

    tabla_abs, tabla_pct = distribucion_por_tipo(df)
    print("Distribución absoluta por tipo:")
    print(tabla_abs, "\n")
    print("Distribución porcentual por tipo:")
    print(tabla_pct.round(2), "\n")

    # Top 10 por tipo, para los tipos más comunes (ajusta la lista según lo que veas en tabla_abs.columns)
    tipos_principales = tabla_abs.sum().sort_values(ascending=False).head(5).index.tolist()
    for tipo in tipos_principales:
        for clase in ["verdadero", "falso"]:
            print(f"\nTop 10 {tipo} - {clase.upper()}")
            print(top_10_por_tipo(df, clase, tipo).to_string())

    # Top 20 global por clase
    for clase in ["verdadero", "falso"]:
        print(f"\nTop 20 global - {clase.upper()}")
        print(top_20_global(df, clase).to_string())

    graficar_distribucion_tipos(tabla_pct)
    graficar_top_entidades(df, "verdadero")
    graficar_top_entidades(df, "falso")
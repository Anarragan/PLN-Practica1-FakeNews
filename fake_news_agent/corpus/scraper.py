import os
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# ---------- Configuración ----------

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
N_DOCS_PER_CLASS = 30 #modificar para obtener mas documentos por clase (75 para llegar a 150)
MIN_PALABRAS = 100
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PALABRAS_FUGA = ["falso", "falsa", "engañoso", "engañosa", "verdadero", "verdadera",
                 "colombiacheck", "chequeo", "chequeamos", "verificamos"] #Eliminado para que el modelo no se entrene con palabras que delaten la clase de la noticia
PATRON_FUGA = re.compile(r"\b(" + "|".join(PALABRAS_FUGA) + r")\b", re.IGNORECASE)


# ---------- Descarga ----------

def obtener_soup(url):
    """Descarga una URL y la parsea. Devuelve None si falla."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    except requests.RequestException as e:
        print(f"Error al obtener {url}: {e}")
        return None


# ---------- Links (una función por fuente) ----------

def links_bbc(paginas):
    """Verdad: links que parezcan artículos."""
    links = []
    for pagina in paginas:
        soup = obtener_soup(pagina)
        if soup is None:
            continue
        for a in soup.find_all("a", href=True):
            url = urljoin(pagina, a["href"])
            if "/articles/" in url and url not in links:
                links.append(url)
    return links


def links_colombiacheck(paginas):
    """Falso: chequeos cuya franja sea 'falso'."""
    links = []
    for pagina in paginas:
        soup = obtener_soup(pagina)
        if soup is None:
            continue
        for franja in soup.find_all(class_=lambda c: c and "franja-falso" in c.lower()):
            tarjeta = franja.find_parent(lambda t: t.find("a", href=lambda h: h and "/chequeos/" in h))
            if tarjeta is None:
                continue
            a = tarjeta.find("a", href=lambda h: h and "/chequeos/" in h)
            url = urljoin(pagina, a["href"])
            if url not in links:
                links.append(url)
    return links


FUENTES = {
    "Verdad": {
        "paginas": ["https://www.bbc.com/mundo"],
        "funcion_links": links_bbc,
    },
    "Falso": {
        "paginas": [f"https://colombiacheck.com/chequeos?page={i}" for i in range(0, 6)],
        "funcion_links": links_colombiacheck,
    },
}


# ---------- Limpieza y extracción ----------

def limpiar(texto):
    """Quita palabras que delatan la clase y normaliza los espacios."""
    texto = PATRON_FUGA.sub("", texto)
    return re.sub(r"\s+", " ", texto).strip()


def extraer_texto(url):
    """Devuelve título + párrafos ya limpios, o '' si falla."""
    soup = obtener_soup(url)
    if soup is None:
        return ""
    titulo = soup.find("h1") or soup.find("h2")
    titulo = limpiar(titulo.get_text()) if titulo else ""
    parrafos = [limpiar(p.get_text()) for p in soup.find_all("p")]
    parrafos = [p for p in parrafos if len(p) > 40]  # se filtra DESPUÉS de limpiar
    return titulo + "\n\n" + "\n".join(parrafos)


# ---------- Guardado ----------

def scraper(etiqueta, config):
    carpeta = os.path.join(BASE_DIR, etiqueta)
    os.makedirs(carpeta, exist_ok=True)

    links = config["funcion_links"](config["paginas"])
    print(f"{etiqueta}: {len(links)} links obtenidos.")

    guardados = 0
    for url in links:
        if guardados >= N_DOCS_PER_CLASS:
            break
        texto = extraer_texto(url)
        if len(texto.split()) < MIN_PALABRAS:
            continue
        guardados += 1
        ruta = os.path.join(carpeta, f"{etiqueta.lower()}_{guardados:03d}.txt")
        with open(ruta, "w", encoding="utf-8") as f:
            f.write(texto)
        print(f"  [{guardados}/{N_DOCS_PER_CLASS}] {url}")
        time.sleep(1)


if __name__ == "__main__":
    for etiqueta, config in FUENTES.items():
        scraper(etiqueta, config)
# Script para visualizar el corpus normalizado, primeros 5 registros
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

pd.set_option("display.max_colwidth", 80)  
pd.set_option("display.width", 150)    

df = pd.read_csv(BASE_DIR / "corpus_normalizado.csv")

if __name__ == "__main__":
    print(df.shape)
    print(df["clase"].value_counts(), "\n")
    print(df[["clase", "texto_original", "texto_normalizado"]].head())
"""
Texto → Tabla → EDA
===================
Pega un párrafo con cifras. Un LLM (Llama 3.3 70B vía Groq) extrae los datos en una tabla
estructurada, la app hace el EDA (perfilado, estadísticas y gráficos) y el modelo interpreta
los resultados en un chat.

Uso:
    pip install -r requirements.txt
    streamlit run app.py

La GROQ API Key se escribe en la barra lateral; no se guarda en ningún archivo.
"""

import io
import json
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq

# ======================================================================================
# Configuración
# ======================================================================================
st.set_page_config(page_title="Texto → Tabla → EDA", page_icon="🔎", layout="wide")

MODELOS = {
    "Llama 3.3 70B (llama-3.3-70b-versatile)": "llama-3.3-70b-versatile",
    "GPT-OSS 120B (openai/gpt-oss-120b)": "openai/gpt-oss-120b",
    "Qwen 3.6 27B (qwen/qwen3.6-27b)": "qwen/qwen3.6-27b",
}

PALETA = px.colors.qualitative.Set2

PROMPT_EXTRACCION = """Eres un extractor de datos. Recibes un texto en lenguaje natural y devuelves
las cifras que contiene como una tabla tidy (una observación por fila).

Devuelve EXCLUSIVAMENTE un objeto JSON con esta forma:
{
  "titulo": "título corto de la tabla",
  "columnas": [{"nombre": "Departamento", "tipo": "texto", "unidad": null},
               {"nombre": "Produccion", "tipo": "numero", "unidad": "toneladas"}],
  "filas": [["Antioquia", 1200], ["Huila", 890]],
  "notas": ["supuestos o ambigüedades relevantes"],
  "cifras_no_extraidas": ["cifras del texto que no encajaron en la tabla"]
}

Reglas:
- "tipo" solo puede ser "texto", "numero" o "fecha" (fechas en formato ISO YYYY-MM-DD).
- Los valores numéricos van como números JSON puros: sin separadores de miles, sin símbolos,
  sin unidades, sin signo de porcentaje. La unidad va en la columna correspondiente.
- Cada fila debe tener exactamente tantos elementos como columnas, en el mismo orden.
- Usa null cuando el texto no da ese dato. No inventes valores ni rellenes con promedios.
- Estructura tidy: si el texto compara entidades (regiones, años, productos), cada entidad
  es una fila y cada métrica una columna. Nunca metas dos métricas en la misma columna.
- Convierte a número los porcentajes (12,5% -> 12.5) e indica la unidad "%".
- Convierte magnitudes escritas en palabras ("2,5 millones" -> 2500000) y anota el supuesto en "notas".
- Nombres de columna cortos, sin espacios ni acentos (usa guion bajo).
- Si el texto no contiene cifras aprovechables, devuelve "filas": [] y explica por qué en "notas".
- No añadas texto fuera del JSON.
"""

SYSTEM_ANALISTA = """Eres un analista de datos. Interpretas el EDA de una tabla que fue extraída
de un texto en lenguaje natural. Respondes en español.

Reglas:
- Usa únicamente las cifras del PERFIL DE DATOS entregado; no inventes números.
- Sé explícito sobre las limitaciones: la tabla suele tener pocas filas, así que evita
  conclusiones estadísticas fuertes y no hables de significancia si n es pequeño.
- Distingue correlación de causalidad; marca las hipótesis como hipótesis.
- Señala vacíos, valores atípicos e inconsistencias que veas en el perfil.
- Concisión: máximo 4 párrafos cortos o 6 viñetas, citando las cifras que respaldan cada punto.
"""

EJEMPLO_TEXTO = (
    "En 2025 la cooperativa reportó ventas por 2.450 millones de pesos, un 12,5% más que en 2024. "
    "El café representó 1.320 millones con 480 toneladas vendidas; el cacao 680 millones con 210 "
    "toneladas y el aguacate 450 millones con 150 toneladas. Antioquia aportó el 45% de las ventas, "
    "Huila el 30% y Nariño el 25%. El costo logístico promedio fue de 180.000 pesos por tonelada y "
    "la cooperativa cerró el año con 1.240 productores afiliados, 90 más que el año anterior."
)

EJEMPLO_TABLA = pd.DataFrame({
    "Producto": ["Café", "Cacao", "Aguacate"],
    "Ventas_millones_COP": [1320.0, 680.0, 450.0],
    "Volumen_toneladas": [480.0, 210.0, 150.0],
})

SUGERENCIAS = [
    "Explícame los hallazgos principales de esta tabla",
    "¿Qué relación hay entre las variables numéricas?",
    "¿Qué valores se ven atípicos o dudosos?",
    "¿Qué información falta para un análisis sólido?",
]


# ======================================================================================
# Extracción con el LLM
# ======================================================================================
def extraer_json(texto_modelo: str) -> dict:
    """Parsea el JSON del modelo, tolerando bloques de código o texto alrededor."""
    limpio = texto_modelo.strip()
    limpio = re.sub(r"^```(?:json)?|```$", "", limpio, flags=re.MULTILINE).strip()
    try:
        return json.loads(limpio)
    except json.JSONDecodeError:
        inicio, fin = limpio.find("{"), limpio.rfind("}")
        if inicio == -1 or fin == -1:
            raise ValueError("El modelo no devolvió un JSON reconocible.")
        return json.loads(limpio[inicio:fin + 1])


def construir_dataframe(spec: dict) -> tuple[pd.DataFrame, dict]:
    """Convierte la especificación JSON del modelo en un DataFrame con tipos correctos."""
    columnas = spec.get("columnas") or []
    filas = spec.get("filas") or []
    if not columnas:
        raise ValueError("El modelo no devolvió columnas.")

    nombres = [c["nombre"] for c in columnas]
    # Descartar filas con longitud distinta al número de columnas
    validas = [f for f in filas if isinstance(f, (list, tuple)) and len(f) == len(nombres)]
    descartadas = len(filas) - len(validas)

    df = pd.DataFrame(validas, columns=nombres)

    for c in columnas:
        nombre, tipo = c["nombre"], (c.get("tipo") or "texto").lower()
        if tipo == "numero":
            df[nombre] = pd.to_numeric(df[nombre], errors="coerce")
        elif tipo == "fecha":
            df[nombre] = pd.to_datetime(df[nombre], errors="coerce")
        else:
            df[nombre] = df[nombre].astype("string")

    meta = {
        "titulo": spec.get("titulo") or "Tabla extraída",
        "unidades": {c["nombre"]: c.get("unidad") for c in columnas},
        "notas": spec.get("notas") or [],
        "no_extraidas": spec.get("cifras_no_extraidas") or [],
        "filas_descartadas": descartadas,
    }
    return df, meta


def llamar_extraccion(texto: str, api_key: str, modelo: str) -> dict:
    cliente = Groq(api_key=api_key)
    r = cliente.chat.completions.create(
        model=modelo,
        messages=[
            {"role": "system", "content": PROMPT_EXTRACCION},
            {"role": "user", "content": f"Texto:\n\"\"\"\n{texto.strip()}\n\"\"\""},
        ],
        temperature=0.0,
        max_tokens=3000,
        response_format={"type": "json_object"},
    )
    return extraer_json(r.choices[0].message.content)


def mensaje_error(e: Exception, modelo: str) -> str:
    msg = str(e)
    if "401" in msg or "authentication" in msg.lower() or "invalid_api_key" in msg:
        return "API Key inválida. Revísala en console.groq.com/keys."
    if "429" in msg or "rate limit" in msg.lower():
        return "Límite de peticiones alcanzado. Espera unos segundos y reintenta."
    if "decommissioned" in msg.lower() or "model_not_found" in msg or "404" in msg:
        return f"El modelo `{modelo}` no está disponible en tu cuenta. Elige otro en la barra lateral."
    return f"Error al llamar a Groq: {msg}"


# ======================================================================================
# EDA
# ======================================================================================
def perfil_texto(df: pd.DataFrame, meta: dict) -> str:
    """Perfil compacto de la tabla, en texto, para que el LLM lo interprete."""
    num = df.select_dtypes(include="number").columns.tolist()
    cat = df.select_dtypes(include=["string", "object", "category"]).columns.tolist()

    p = [
        f"### {meta.get('titulo', 'Tabla')}",
        f"- Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas",
        f"- Columnas numéricas: {', '.join(num) if num else 'ninguna'}",
        f"- Columnas categóricas: {', '.join(cat) if cat else 'ninguna'}",
        f"- Unidades declaradas: {meta.get('unidades', {})}",
        f"- Celdas vacías por columna: {df.isna().sum().to_dict()}",
        f"- Filas duplicadas: {int(df.duplicated().sum())}",
    ]
    if meta.get("notas"):
        p.append(f"- Notas del extractor: {' | '.join(map(str, meta['notas']))}")
    if meta.get("no_extraidas"):
        p.append(f"- Cifras del texto NO incluidas en la tabla: {' | '.join(map(str, meta['no_extraidas']))}")

    p.append("\n### Tabla completa\n" + df.to_markdown(index=False))

    if num:
        p.append("\n### Estadísticas descriptivas\n" + df[num].describe().round(3).to_markdown())
    if len(num) >= 2:
        p.append("\n### Correlaciones de Pearson\n" + df[num].corr(numeric_only=True).round(3).to_markdown())
    for c in cat:
        vc = df[c].value_counts(dropna=False)
        if len(vc) <= 15:
            p.append(f"\n### Frecuencias de {c}\n{vc.to_markdown()}")
    return "\n".join(p)


def graficos_eda(df: pd.DataFrame, meta: dict) -> None:
    num = df.select_dtypes(include="number").columns.tolist()
    cat = df.select_dtypes(include=["string", "object", "category"]).columns.tolist()
    fecha = df.select_dtypes(include="datetime").columns.tolist()

    if not num:
        st.info("La tabla no tiene columnas numéricas: no hay gráficos que generar.", icon="📉")
        return

    def etiqueta(col: str) -> str:
        u = (meta.get("unidades") or {}).get(col)
        return f"{col} ({u})" if u else col

    # Serie temporal si hay fechas
    if fecha:
        eje = fecha[0]
        for c in num:
            g = df.dropna(subset=[eje, c]).sort_values(eje)
            if len(g) >= 2:
                fig = px.line(g, x=eje, y=c, markers=True,
                              labels={c: etiqueta(c)}, title=f"{c} en el tiempo")

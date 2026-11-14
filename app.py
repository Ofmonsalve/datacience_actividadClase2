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
from pathlib import Path

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


LIBRO_FLOTA = Path(__file__).parent / "drones_ucrania.xlsx"

UNIDADES_FLOTA = {
    "Autonomia_Min": "minutos",
    "Alcance_km": "km",
    "Misiones_Mensuales": "misiones",
    "Horas_Vuelo_Acumuladas": "horas",
    "Tasa_Exito_Mision_Pct": "%",
    "Coste_Unitario_USD": "USD",
    "Coste_por_Mision_USD": "USD",
}


@st.cache_data(show_spinner=False)
def cargar_flota(ruta: str) -> pd.DataFrame:
    """Lee la hoja Datos del libro de la actividad."""
    df = pd.read_excel(ruta, sheet_name="Datos")
    df["Fecha_Alta"] = pd.to_datetime(df["Fecha_Alta"], errors="coerce")
    return df


def resumen_flota(df: pd.DataFrame) -> str:
    """Perfil agregado, sin volcar las 500 filas, para la pantalla y para el analista."""
    num = [
        "Autonomia_Min", "Alcance_km", "Misiones_Mensuales", "Horas_Vuelo_Acumuladas",
        "Tasa_Exito_Mision_Pct", "Coste_Unitario_USD", "Coste_por_Mision_USD",
    ]
    num = [c for c in num if c in df.columns]
    partes = [
        "### Flota filtrada (datos sintéticos)",
        f"- Aeronaves: {len(df)}",
        f"- Clases: {', '.join(sorted(df['Clase'].dropna().unique()))}",
        f"- Estados: {df['Estado'].value_counts().to_dict()}",
        f"- Tasa de éxito media: {df['Tasa_Exito_Mision_Pct'].mean():.1f}%",
        f"- Alcance medio: {df['Alcance_km'].mean():.1f} km",
        f"- Autonomía media: {df['Autonomia_Min'].mean():.1f} min",
        f"- Coste unitario medio: {df['Coste_Unitario_USD'].mean():,.0f} USD",
    ]
    if num:
        partes.append("\n### Medias por clase\n" + df.groupby("Clase", observed=True)[num].mean().round(1).to_markdown())
        partes.append("\n### Conteos por sector\n" + df["Sector_Frente"].value_counts().to_markdown())
        if len(num) >= 2:
            partes.append("\n### Correlaciones de Pearson\n" + df[num].corr(numeric_only=True).round(3).to_markdown())
    return "\n".join(partes)


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

    cats_utiles = [c for c in cat if 1 < df[c].nunique(dropna=True) <= 12]

    if fecha:
        eje = fecha[0]
        for c in num[:2]:
            g = df.dropna(subset=[eje, c]).sort_values(eje)
            if len(g) >= 2:
                fig = px.line(g, x=eje, y=c, markers=True,
                              labels={c: etiqueta(c)}, title=f"{c} en el tiempo")
                fig.update_layout(height=360, margin=dict(t=50))
                st.plotly_chart(fig, width="stretch")

    columnas = st.columns(2)
    if cats_utiles:
        c0, n0 = cats_utiles[0], num[0]
        g = df.groupby(c0, as_index=False, observed=True)[n0].mean().sort_values(n0)
        fig = px.bar(g, x=n0, y=c0, orientation="h",
                     labels={n0: etiqueta(n0), c0: ""},
                     title=f"{etiqueta(n0)} medio por {c0}")
        fig.update_layout(height=380, margin=dict(t=50))
        columnas[0].plotly_chart(fig, width="stretch")
    if len(num) >= 1:
        fig = px.box(df, y=num[0], labels={num[0]: etiqueta(num[0])},
                     title=f"Distribución de {etiqueta(num[0])}")
        fig.update_layout(height=380, margin=dict(t=50))
        columnas[1 if cats_utiles else 0].plotly_chart(fig, width="stretch")

    if len(num) >= 2:
        fig = px.scatter(
            df, x=num[0], y=num[1],
            color=cats_utiles[0] if cats_utiles else None,
            labels={num[0]: etiqueta(num[0]), num[1]: etiqueta(num[1])},
            title=f"{etiqueta(num[0])} vs {etiqueta(num[1])}",
        )
        fig.update_layout(height=420, margin=dict(t=50))
        st.plotly_chart(fig, width="stretch")

    if len(num) >= 2:
        corr = df[num].corr(numeric_only=True).round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu",
                        zmin=-1, zmax=1, title="Correlaciones")
        fig.update_layout(height=420, margin=dict(t=50))
        st.plotly_chart(fig, width="stretch")


def llamar_analista(pregunta: str, perfil: str, api_key: str, modelo: str) -> str:
    cliente = Groq(api_key=api_key)
    r = cliente.chat.completions.create(
        model=modelo,
        messages=[
            {"role": "system", "content": SYSTEM_ANALISTA + "\n\nPERFIL DE DATOS:\n" + perfil},
            {"role": "user", "content": pregunta},
        ],
        temperature=0.2,
        max_tokens=1200,
    )
    return r.choices[0].message.content


def pintar_flota(df: pd.DataFrame) -> None:
    """Tablero de la actividad: el libro drones_ucrania.xlsx ya cargado."""
    k = st.columns(5)
    k[0].metric("Aeronaves", f"{len(df):,}")
    k[1].metric("Tasa de éxito", f"{df['Tasa_Exito_Mision_Pct'].mean():.1f}%")
    k[2].metric("Alcance medio", f"{df['Alcance_km'].mean():.1f} km")
    k[3].metric("Autonomía media", f"{df['Autonomia_Min'].mean():.0f} min")
    k[4].metric("Coste medio", f"{df['Coste_Unitario_USD'].mean():,.0f} USD")

    c1, c2 = st.columns(2)
    conteo = df["Clase"].value_counts().rename_axis("Clase").reset_index(name="Aeronaves")
    fig = px.bar(conteo, x="Aeronaves", y="Clase", orientation="h", color="Clase",
                 color_discrete_sequence=PALETA, title="Aeronaves por clase")
    fig.update_layout(height=420, showlegend=False, margin=dict(t=50))
    c1.plotly_chart(fig, width="stretch")

    exito = (df.groupby("Clase", as_index=False, observed=True)["Tasa_Exito_Mision_Pct"]
               .mean().sort_values("Tasa_Exito_Mision_Pct"))
    fig = px.bar(exito, x="Tasa_Exito_Mision_Pct", y="Clase", orientation="h",
                 color="Clase", color_discrete_sequence=PALETA,
                 labels={"Tasa_Exito_Mision_Pct": "Tasa de éxito media (%)"},
                 title="Éxito de misión por clase")
    fig.update_layout(height=420, showlegend=False, margin=dict(t=50))
    c2.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    fig = px.box(df, x="Clase", y="Alcance_km", color="Clase", color_discrete_sequence=PALETA,
                 labels={"Alcance_km": "Alcance (km)", "Clase": ""},
                 title="Alcance por clase")
    fig.update_layout(height=420, showlegend=False, margin=dict(t=50))
    c3.plotly_chart(fig, width="stretch")

    fig = px.scatter(df, x="Autonomia_Min", y="Alcance_km", color="Clase",
                     hover_name="ID_Dron", opacity=0.75, color_discrete_sequence=PALETA,
                     labels={"Autonomia_Min": "Autonomía (min)", "Alcance_km": "Alcance (km)"},
                     title="Autonomía y alcance")
    fig.update_layout(height=420, margin=dict(t=50))
    c4.plotly_chart(fig, width="stretch")

    altas = (df.dropna(subset=["Fecha_Alta"])
               .assign(Mes=lambda d: d["Fecha_Alta"].dt.to_period("M").dt.to_timestamp())
               .groupby("Mes", as_index=False).size())
    fig = px.line(altas, x="Mes", y="size", markers=True,
                  labels={"size": "Altas", "Mes": ""},
                  title="Entradas en servicio por mes")
    fig.update_layout(height=360, margin=dict(t=50))
    st.plotly_chart(fig, width="stretch")

    st.dataframe(df, width="stretch", hide_index=True, height=360)


def pintar_texto(api_key: str, modelo: str) -> None:
    st.markdown(
        "Pega un párrafo con cifras. Con una API Key de Groq, el modelo arma la tabla. "
        "Sin clave, puedes cargar la tabla de ejemplo que ya trae la actividad."
    )
    texto = st.text_area("Texto", value=EJEMPLO_TEXTO, height=160)
    b1, b2 = st.columns(2)
    extraer = b1.button("Extraer tabla con Groq", width="stretch", type="primary")
    ejemplo = b2.button("Cargar tabla de ejemplo", width="stretch")

    if extraer:
        if not api_key:
            st.warning("Escribe la GROQ API Key en la barra lateral para extraer el texto.", icon="🔑")
        elif not texto.strip():
            st.warning("El texto está vacío.")
        else:
            with st.spinner("Extrayendo la tabla…"):
                try:
                    spec = llamar_extraccion(texto, api_key, modelo)
                    df, meta = construir_dataframe(spec)
                    st.session_state.tabla = df
                    st.session_state.meta_tabla = meta
                except Exception as e:
                    st.error(mensaje_error(e, modelo))

    if ejemplo:
        st.session_state.tabla = EJEMPLO_TABLA.copy()
        st.session_state.meta_tabla = {
            "titulo": "Ventas de la cooperativa (ejemplo)",
            "unidades": {"Ventas_millones_COP": "millones COP", "Volumen_toneladas": "toneladas"},
            "notas": ["Tabla de ejemplo incluida en la app; no vino de Groq."],
            "cifras_no_extraidas": [],
        }

    df = st.session_state.get("tabla")
    meta = st.session_state.get("meta_tabla")
    if df is None or meta is None:
        return

    st.subheader(meta.get("titulo") or "Tabla")
    if meta.get("notas"):
        st.caption(" · ".join(map(str, meta["notas"])))
    st.dataframe(df, width="stretch", hide_index=True)
    with st.expander("Perfil de la tabla"):
        st.markdown(perfil_texto(df, meta))
    graficos_eda(df, meta)


def pintar_analista(perfil: str, api_key: str, modelo: str) -> None:
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if not api_key:
        st.info("Escribe tu GROQ API Key en la barra lateral para hablar con el analista.", icon="🔑")

    if st.button("Limpiar chat"):
        st.session_state.chat = []
        st.rerun()

    pregunta_click = None
    if not st.session_state.chat:
        cols = st.columns(2)
        for i, s in enumerate(SUGERENCIAS):
            if cols[i % 2].button(s, key=f"sug_{i}", width="stretch"):
                pregunta_click = s

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    entrada = st.chat_input("Pregunta sobre la flota filtrada")
    pregunta = entrada or pregunta_click
    if not pregunta:
        return
    if not api_key:
        st.warning("Falta la API Key en la barra lateral.", icon="⚠️")
        return

    with st.chat_message("user"):
        st.markdown(pregunta)
    with st.chat_message("assistant"):
        try:
            texto = llamar_analista(pregunta, perfil, api_key, modelo)
        except Exception as e:
            st.error(mensaje_error(e, modelo))
            return
    st.session_state.chat.append({"role": "user", "content": pregunta})
    st.session_state.chat.append({"role": "assistant", "content": texto})
    st.rerun()


# ======================================================================================
# Pantalla
# ======================================================================================
st.title("Texto → Tabla → EDA")
st.caption("Actividad en clase · el libro de la flota se carga solo desde `drones_ucrania.xlsx`.")

with st.sidebar:
    st.header("Analista IA")
    api_key = st.text_input("GROQ API Key", type="password", placeholder="gsk_...",
                            help="Solo vive en esta sesión. No se guarda en ningún archivo.")
    modelo = MODELOS[st.selectbox("Modelo", list(MODELOS))]
    st.caption("La clave se pide en console.groq.com/keys. Sin ella puedes ver la flota y la tabla de ejemplo.")
    st.divider()
    st.header("Filtros de la flota")

if not LIBRO_FLOTA.exists():
    st.error(f"No está el libro `{LIBRO_FLOTA.name}` junto a `app.py`.")
    st.stop()

try:
    flota = cargar_flota(str(LIBRO_FLOTA))
except Exception as e:
    st.error(f"No pude leer el libro de la actividad: {e}")
    st.stop()

with st.sidebar:
    clases = st.multiselect("Clase", sorted(flota["Clase"].unique()), default=sorted(flota["Clase"].unique()))
    estados = st.multiselect("Estado", sorted(flota["Estado"].unique()), default=sorted(flota["Estado"].unique()))
    sectores = st.multiselect("Sector", sorted(flota["Sector_Frente"].unique()), default=sorted(flota["Sector_Frente"].unique()))
    interferencia = st.multiselect(
        "Interferencia GNSS", sorted(flota["Interferencia_GNSS"].unique()),
        default=sorted(flota["Interferencia_GNSS"].unique()),
    )

vista = flota[
    flota["Clase"].isin(clases)
    & flota["Estado"].isin(estados)
    & flota["Sector_Frente"].isin(sectores)
    & flota["Interferencia_GNSS"].isin(interferencia)
].copy()
st.sidebar.markdown(f"**{len(vista):,} de {len(flota):,} aeronaves**")

tab_flota, tab_texto, tab_chat = st.tabs(["Flota de drones", "Texto → tabla", "Analista"])

with tab_flota:
    st.warning(
        "Los 500 registros son sintéticos (simulación estadística, semilla 2026). "
        "No describen unidades ni operaciones reales. Sirven para practicar el EDA.",
        icon="🧪",
    )
    if vista.empty:
        st.warning("Ninguna aeronave cumple los filtros.")
    else:
        pintar_flota(vista)
        with st.expander("Resumen que vería el analista"):
            st.markdown(resumen_flota(vista))

with tab_texto:
    pintar_texto(api_key, modelo)

with tab_chat:
    if vista.empty:
        st.warning("Ajusta los filtros: el analista lee la flota filtrada.")
    else:
        pintar_analista(resumen_flota(vista), api_key, modelo)

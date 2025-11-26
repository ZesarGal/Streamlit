import streamlit as st
import calendar
from datetime import date
import pandas as pd

st.title("Calendario – Fechas importantes (Nov 2025 – Ene 2026)")

# Inicializar diccionario de eventos en la sesión
if "events" not in st.session_state:
    # Clave: "YYYY-MM-DD", Valor: texto de la actividad
    st.session_state["events"] = {}

# --- Formulario para agregar/editar eventos ---
st.subheader("Agregar / editar fecha")

col1, col2 = st.columns(2)

with col1:
    selected_date = st.date_input(
        "Elige una fecha",
        value=date(2025, 11, 1),
        min_value=date(2025, 11, 1),
        max_value=date(2026, 1, 31),
    )

with col2:
    key = selected_date.strftime("%Y-%m-%d")
    existing_text = st.session_state["events"].get(key, "")
    activity = st.text_input("Actividad para ese día", value=existing_text)

if st.button("Guardar fecha"):
    key = selected_date.strftime("%Y-%m-%d")
    st.session_state["events"][key] = activity
    st.success(f"Guardado: {key} → {activity}")

st.markdown("---")

# --- Función para construir un DataFrame tipo calendario ---
def build_month_df(year: int, month: int) -> pd.DataFrame:
    cal = calendar.Calendar(firstweekday=0)  # 0 = lunes
    weeks = cal.monthdayscalendar(year, month)

    data = []
    for week in weeks:
        row = []
        for day in week:
            if day == 0:
                row.append("")  # celda vacía
            else:
                row.append(str(day))
        data.append(row)

    df = pd.DataFrame(
        data,
        columns=["L", "M", "X", "J", "V", "S", "D"]
    )
    return df

def show_month(title: str, year: int, month: int):
    st.markdown(f"#### {title}")
    df = build_month_df(year, month)

    # Estilo: círculo rojo alrededor de las fechas con evento,
    # sin modificar width/height ni inline-block para evitar que se encimen.
    def style_events(val):
        if val == "":
            return ""
        if not val.isdigit():
            return ""
        day = int(val)
        key = f"{year}-{month:02d}-{day:02d}"
        if key in st.session_state["events"]:
            return (
                "color: red;"
                "border: 2px solid red;"
                "border-radius: 50%;"
            )
        return ""

    styler = (
        df.style
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("text-align", "center"),
                    ("padding", "0.2rem"),
                    ("font-size", "0.8rem"),
                ],
            },
            {
                "selector": "td",


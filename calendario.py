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
                row.append("")
            else:
                key = f"{year}-{month:02d}-{day:02d}"
                if key in st.session_state["events"]:
                    # Día con evento → marcar con un punto
                    row.append(f"{day} •")
                else:
                    row.append(str(day))
        data.append(row)

    df = pd.DataFrame(
        data,
        columns=["L", "M", "X", "J", "V", "S", "D"]  # <-- aquí el cambio
    )
    return df

# --- Mostrar calendarios de noviembre, diciembre y enero ---
st.subheader("Calendario")

col_nov, col_dic, col_ene = st.columns(3)

with col_nov:
    st.markdown("#### Noviembre 2025")
    st.table(build_month_df(2025, 11))

with col_dic:
    st.markdown("#### Diciembre 2025")
    st.table(build_month_df(2025, 12))

with col_ene:
    st.markdown("#### Enero 2026")
    st.table(build_month_df(2026, 1))

st.caption("Días con evento están marcados como `número •`.")

st.markdown("---")

# --- Lista de eventos guardados ---
st.subheader("Fechas marcadas")

if st.session_state["events"]:
    for key, text in sorted(st.session_state["events"].items()):
        st.write(f"📅 **{key}** → {text}")
else:
    st.write("Aún no has marcado ninguna fecha.")


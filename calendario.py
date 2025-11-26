import streamlit as st
import calendar
from datetime import date

st.title("Calendario 2025 – Fechas importantes (Nov–Dic)")

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
        max_value=date(2025, 12, 31),
    )

with col2:
    existing_text = st.session_state["events"].get(
        selected_date.strftime("%Y-%m-%d"), ""
    )
    activity = st.text_input("Actividad para ese día", value=existing_text)

if st.button("Guardar fecha"):
    key = selected_date.strftime("%Y-%m-%d")
    st.session_state["events"][key] = activity
    st.success(f"Guardado: {key} → {activity}")

st.markdown("---")

# --- Función para dibujar calendario sencillo marcando días con evento ---
def draw_month(year, month):
    st.markdown(f"### {calendar.month_name[month]} {year}")
    cal = calendar.Calendar(firstweekday=0)  # lunes = 0 (si quieres)
    
    st.text("Lu Ma Mi Ju Vi Sa Do")
    for week in cal.monthdayscalendar(year, month):
        line = ""
        for day in week:
            if day == 0:
                line += "   "
            else:
                key = f"{year}-{month:02d}-{day:02d}"
                if key in st.session_state["events"]:
                    # Día con evento → en negritas
                    line += f"{day:2d}* "
                else:
                    line += f"{day:2d}  "
        st.text(line)

# --- Mostrar calendarios de noviembre y diciembre con días marcados ---
col_nov, col_dic = st.columns(2)

with col_nov:
    draw_month(2025, 11)

with col_dic:
    draw_month(2025, 12)

st.markdown("---")

# --- Lista de eventos guardados ---
st.subheader("Fechas marcadas")

if st.session_state["events"]:
    for key, text in sorted(st.session_state["events"].items()):
        st.write(f"📅 **{key}** → {text}")
else:
    st.write("Aún no has marcado ninguna fecha.")


import streamlit as st
import datetime as dt
import calendar

# Configuración básica
st.set_page_config(page_title="Calendario 2025", layout="centered")

st.title("📅 Calendario 2025 con actividades")
st.write(
    "Haz clic en un día para seleccionarlo y escribe la **actividad** que vas a realizar. "
    "El año está fijado en **2025**."
)

year = 2025

# --- Estado inicial ---
# Diccionario: "YYYY-MM-DD" -> texto de actividad
if "events_2025" not in st.session_state:
    st.session_state.events_2025 = {}

# Fecha seleccionada actualmente
if "selected_date" not in st.session_state:
    st.session_state.selected_date = None

# --- Controles en la barra lateral ---
st.sidebar.header("Opciones")

# Atajos para noviembre y diciembre
mes_rapido = st.sidebar.radio(
    "Atajos de mes",
    ["Ninguno", "Noviembre 2025", "Diciembre 2025"],
    index=0
)

if mes_rapido == "Noviembre 2025":
    default_month = 11
elif mes_rapido == "Diciembre 2025":
    default_month = 12
else:
    default_month = 11  # noviembre por defecto

month = st.sidebar.selectbox(
    "Selecciona un mes de 2025",
    options=list(range(1, 13)),
    index=default_month - 1,
    format_func=lambda m: calendar.month_name[m]
)

# Botón para limpiar todas las actividades
if st.sidebar.button("🧹 Borrar todas las actividades"):
    st.session_state.events_2025 = {}
    st.session_state.selected_date = None

# --- Mostrar calendario del mes seleccionado ---
st.subheader(f"{calendar.month_name[month]} {year}")

# Encabezados de días
weekdays = ["Lu", "Ma", "Mi", "Ju", "Vi", "Sa", "Do"]
cols_header = st.columns(7)
for i, wd in enumerate(weekdays):
    cols_header[i].markdown(f"**{wd}**")

# Cuerpo del calendario
for week in calendar.monthcalendar(year, month):
    cols = st.columns(7)
    for i, day in enumerate(week):
        if day == 0:
            cols[i].write(" ")
        else:
            current_date = dt.date(year, month, day)
            date_str = current_date.isoformat()  # "YYYY-MM-DD"

            # Ver si ya hay actividad para ese día
            has_event = date_str in st.session_state.events_2025
            label = f"{day} 📌" if has_event else str(day)

            # Cada día es un botón que selecciona la fecha
            if cols[i].button(label, key=f"btn-{date_str}"):
                st.session_state.selected_date = date_str

# --- Editor de actividad para la fecha seleccionada ---
st.markdown("---")
st.subheader("✍️ Actividad del día")

if st.session_state.selected_date is not None:
    date_str = st.session_state.selected_date
    date_obj = dt.date.fromisoformat(date_str)

    st.markdown(
        f"**Fecha seleccionada:** {date_obj.strftime('%d de %B de %Y')}"
    )

    # Texto actual (si ya había actividad)
    current_text = st.session_state.events_2025.get(date_str, "")

    actividad = st.text_area(
        "Escribe la actividad para este día:",
        value=current_text,
        key=f"text-{date_str}",
        height=100
    )

    cols_actions = st.columns(2)
    with cols_actions[0]:
        if st.button("💾 Guardar actividad", key=f"save-{date_str}"):
            if actividad.strip():
                st.session_state.events_2025[date_str] = actividad.strip()
                st.success("Actividad guardada.")
            else:
                # Si el texto está vacío, borramos la actividad
                st.session_state.events_2025.pop(date_str, None)
                st.info("Actividad eliminada (texto vacío).")

    with cols_actions[1]:
        if st.button("🗑️ Eliminar actividad de este día", key=f"del-{date_str}"):
            st.session_state.events_2025.pop(date_str, None)
            st.session_state.selected_date = None
            st.info("Actividad eliminada.")
else:
    st.info("Selecciona un día en el calendario para agregar una actividad.")

# --- Listado de todas las actividades ---
st.markdown("---")
st.subheader("📋 Resumen de actividades 2025")

if st.session_state.events_2025:
    for date_str in sorted(st.session_state.events_2025.keys()):
        d = dt.date.fromisoformat(date_str)
        st.markdown(
            f"- **{d.strftime('%d de %B de %Y')}**: {st.session_state.events_2025[date_str]}"
        )
else:
    st.write("Aún no has registrado actividades.")

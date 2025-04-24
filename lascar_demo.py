import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import altair as alt

st.set_page_config(page_title="Lascar Monitoring", layout="wide")

# -----------------------------
# Helper functions & fake data
# -----------------------------
np.random.seed(42)
# Usar 'min' en lugar de 'T' que está obsoleto
DATE_RANGE = pd.date_range(end=dt.datetime.now(dt.timezone.utc), periods=288, freq="5min")  # last 24 h, 5‑min res


def fake_station_df(name: str, peak_w: int, init_soc: float) -> pd.DataFrame:
    """Return fake dataframe for a station."""
    df = pd.DataFrame(index=DATE_RANGE)
    n_points = len(df)
    
    # --- Existing Solar & Battery ---
    solar_curve = np.sin(np.linspace(-np.pi / 2, 3 * np.pi / 2, n_points))  # sunrise‑sunset
    solar_noise = np.random.normal(0, peak_w * 0.05, n_points)
    df["solar_W"] = np.clip(peak_w * solar_curve + solar_noise, 0, None)
    intervalo_horas = 5 / 60 
    df["solar_Wh"] = df["solar_W"] * intervalo_horas
    consumo_promedio_W = peak_w * 0.15
    energia_neta_Wh = df["solar_Wh"] - (consumo_promedio_W * intervalo_horas)
    capacidad_bateria_Wh = peak_w * 8 
    soc_variation = np.cumsum(energia_neta_Wh) / capacidad_bateria_Wh * 100
    df["battery_soc"] = np.clip(init_soc + soc_variation + np.random.normal(0, 0.8, n_points), 0, 100)

    # --- New Environmental & Network Data ---
    # Temperature (°C) - Varies sinusoidally daily, with some noise
    temp_base = 5 + np.random.uniform(-2, 2) # Base temp varies slightly per station
    temp_amplitude = 10 + np.random.uniform(-1, 1)
    temp_curve = np.sin(np.linspace(0, 2 * np.pi, n_points)) # Daily cycle
    df["temperature_C"] = temp_base + temp_amplitude * temp_curve + np.random.normal(0, 0.5, n_points)

    # Humidity (%) - Tends to be inverse to temperature, with noise
    humidity_base = 40 + np.random.uniform(-5, 5)
    humidity_variation = -15 * temp_curve # Inverse relation to temp
    df["humidity_pct"] = np.clip(humidity_base + humidity_variation + np.random.normal(0, 3, n_points), 5, 95)

    # Air Quality (PM2.5 µg/m³) - Base level with random spikes (e.g., volcanic ash)
    pm25_base = 8 + np.random.uniform(-3, 3)
    spikes = np.random.choice([0, 50, 100], size=n_points, p=[0.98, 0.015, 0.005]) # Random spikes
    df["pm25_ug_m3"] = np.clip(pm25_base + np.random.normal(0, 2, n_points) + spikes, 0, None)

    # Internet Quality (Ping ms) - Base level with random higher latency periods
    ping_base = 50 + np.random.uniform(-10, 20)
    high_latency_periods = np.random.choice([0, 150], size=n_points, p=[0.95, 0.05])
    df["ping_ms"] = ping_base + np.random.normal(0, 10, n_points) + high_latency_periods
    df["ping_ms"] = np.clip(df["ping_ms"], 20, None) # Min ping 20ms

    df.reset_index(inplace=True)
    df.rename(columns={"index": "timestamp"}, inplace=True)
    df["station"] = name
    return df

# -----------------------------
# Data Loading / Generation
# -----------------------------
STATIONS = {
    "UVcam": fake_station_df("UVcam", peak_w=160, init_soc=75),
    "Sismico 1": fake_station_df("Sismico 1", peak_w=100, init_soc=70),
    "Sismico 2": fake_station_df("Sismico 2", peak_w=100, init_soc=80),
}

ALL_DF = pd.concat(STATIONS.values(), ignore_index=True)

# -----------------------------
# Sidebar navigation
# -----------------------------
PAGES = ["Dashboard", "UVcam", "Sismico 1", "Sismico 2"]
page = st.sidebar.radio("📍 Select view", PAGES)

# Función para estilo de fondo (usando gradientes)
def set_bg_hack():
    # Gradiente oscuro con tonos rojos/naranjas
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: linear-gradient(to bottom, #1a0000, #330000, #4d0000);
             /* O alternativamente, un gradiente más tipo atardecer volcánico: */
             /* background: linear-gradient(to bottom, #2c1a1d, #5a2a2a, #e87a5a, #fcab78); */
             background-attachment: fixed;
         }}
         /* Ajustar color de texto principal para mejor contraste en fondo oscuro */
         body, .stApp, .stMarkdown, .stMetric, .stDataFrame, .stExpander, .stRadio > label, .stButton > button {{
             /* color: #f0f0f0; */ /* Color de texto claro - ANTERIOR */
             color: #ffffff !important; /* Blanco puro para texto general */
         }}
         /* Mejorar contraste de títulos y subtítulos (ya son blancos) */
         h1, h2, h3, h4, h5, h6 {{
             color: #ffffff; 
         }}
         /* Estilo de métricas */
         /* Selector para etiqueta de métrica usando data-testid */
         [data-testid="stMetricLabel"] {{
            color: #ffffff !important; /* Blanco para etiquetas */
            opacity: 0.85; /* Ligeramente menos opaco que el valor para diferenciar (opcional) */
         }}
         /* Selector para valor de métrica usando data-testid */
         [data-testid="stMetricValue"] {{
            color: #ffffff !important; /* Blanco para valores */
         }}
         /* Selector para delta de métrica (opcional, por si acaso) */
         [data-testid="stMetricDelta"] {{
            color: #a0ffa0 !important; /* Un verde claro para deltas positivos */
         }}
         [data-testid="stMetricDelta"] > div[data-delta-style="inverse"] {{
             color: #ffa0a0 !important; /* Un rojo claro para deltas negativos */
         }}
         /* Estilo sidebar (opcional, para que combine) */
         section[data-testid="stSidebar"] {{
             background-color: rgba(40, 40, 40, 0.85) !important;
             backdrop-filter: blur(5px);
         }}
          section[data-testid="stSidebar"] * {{
              /* color: #f0f0f0 !important; */ /* Texto claro en sidebar - ANTERIOR */
              color: #ffffff !important; /* Blanco puro en sidebar */
          }}
         /* Asegurar que los botones también tengan texto blanco */
         .stButton > button {{
             color: #ffffff !important; 
         }}

         /* Links (opcionalmente blancos también) */
         a:link, a:visited {{
             color: #ffffff; /* Blanco para links */
             /* text-decoration: none; */ /* Descomentar para quitar subrayado */
         }}
         a:hover, a:active {{
              color: #dddddd; /* Un gris muy claro al pasar el mouse */
             /* text-decoration: underline; */
         }}
         
         /* --- NUEVO: Filtro para el logo CKELAR --- */
         /* Target img tag whose src contains the logo filename */
         img[src*="logo-ckelar-web.png"] {{
             filter: brightness(0) invert(1); 
             /* Otros filtros alternativos por si el anterior no funciona bien:
             filter: grayscale(1) brightness(100); 
             filter: contrast(0) brightness(2) grayscale(1) invert(0);
             */
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Llamar a la función para aplicar el fondo al inicio
set_bg_hack()

# --- Header con Logo y Título --- 
logo_url = "https://ckelar.org/wp-content/themes/ckelar/img/logo-ckelar-web.png"

# Crear columnas para el header
col_logo, col_title = st.columns([1, 5]) # Ajusta la proporción si es necesario

with col_logo:
    st.image(logo_url, width=225) # Ancho aumentado (150 * 1.5)

with col_title:
    # El título específico de la página se mostrará más abajo
    # Aquí podemos poner un título general o dejarlo vacío
    # st.title("Monitoreo Lascar") # Opcional: Título general aquí
    pass # Dejar espacio para el título específico de la página

st.divider() # Separador visual
# --- Fin Header --- 

# -----------------------------
# Dashboard Page – aggregated view
# -----------------------------
if page == "Dashboard":
    # El título ahora va debajo del header
    st.title("🌋 Lascar Volcano – Power & Telemetry Dashboard (Demo)")
    st.markdown("Vista agregada de las estaciones de monitoreo.")

    # --- Métricas Agregadas --- 
    st.subheader("📊 Overall Status (Last 24h / Latest)")
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

    # Total generation last 24 h
    gen_last24_kwh = ALL_DF["solar_Wh"].sum() / 1000 
    m_col1.metric(label="⚡ Total Solar Gen.", value=f"{gen_last24_kwh:,.1f} kWh")

    # Average SOC (last value for each station)
    latest_readings = ALL_DF.loc[ALL_DF.groupby("station")["timestamp"].idxmax()]
    soc_avg = latest_readings["battery_soc"].mean()
    m_col2.metric(label="🔋 Avg Battery SOC", value=f"{soc_avg:.0f}%", help="Promedio del último SOC reportado.")

    # Average Temp (latest)
    temp_avg = latest_readings["temperature_C"].mean()
    m_col3.metric(label="🌡️ Avg Temp.", value=f"{temp_avg:.1f} °C")

    # Average Humidity (latest)
    hum_avg = latest_readings["humidity_pct"].mean()
    m_col4.metric(label="💧 Avg Humidity", value=f"{hum_avg:.0f}% ")
    
    # Max PM2.5 (latest) - Podría ser más útil que el promedio
    pm25_max = latest_readings["pm25_ug_m3"].max()
    m_col5.metric(label="💨 Max PM2.5", value=f"{pm25_max:.0f} µg/m³", help="Máximo valor de PM2.5 reportado recientemente.")
    
    st.divider()
    
    # --- Gráficos y Tabla --- 
    col1, col2 = st.columns([2, 1]) # Dar más espacio al gráfico de área

    with col1:
        # Stacked area de producción solar
        st.subheader("☀️ Solar Power Generation")
        chart_solar = (
            alt.Chart(ALL_DF)
            .mark_area(opacity=0.7)
            .encode(
                x=alt.X("timestamp:T", title="Time"),
                y=alt.Y("solar_W:Q", stack="zero", title="Solar Power (W)"), 
                color=alt.Color("station:N", title="Station"),
                order=alt.Order("station", sort="ascending"),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Time"),
                    alt.Tooltip("station:N", title="Station"), 
                    alt.Tooltip("solar_W:Q", title="Power (W)", format=".0f")
                ]
            )
            .interactive()
        )
        st.altair_chart(chart_solar, use_container_width=True)

    with col2:
         # Gráfico de barras SOC actual
        st.subheader("📊 Current Battery")
        soc_bars = alt.Chart(latest_readings).mark_bar().encode(
            x=alt.X('battery_soc', title="SOC (%)"),
            y=alt.Y('station', sort='-x', title="Station"),
            color=alt.Color('battery_soc', scale=alt.Scale(range=['#e74c3c', '#f1c40f', '#2ecc71']), legend=None),
            tooltip=['station', alt.Tooltip('battery_soc', format=".0f") ]
        ).properties(
            height=250 # Ajustar altura si es necesario
        )
        st.altair_chart(soc_bars, use_container_width=True)
        
        # Indicador simple de calidad de red (basado en ping promedio)
        st.subheader("🌐 Network Status")
        avg_ping = latest_readings["ping_ms"].mean()
        if avg_ping < 80:
            st.success(f"Avg Ping: {avg_ping:.0f} ms (Good)")
        elif avg_ping < 150:
            st.warning(f"Avg Ping: {avg_ping:.0f} ms (Fair)")
        else:
            st.error(f"Avg Ping: {avg_ping:.0f} ms (Poor)")
            

    # Tabla Resumen (movida abajo)
    st.divider()
    st.subheader("📈 Latest Readings Summary")
    st.dataframe(
        latest_readings[["station", "timestamp", "solar_W", "battery_soc", "temperature_C", "humidity_pct", "pm25_ug_m3", "ping_ms"]].rename(
            columns={
                "station": "Station", "timestamp": "Last Update", "solar_W": "Solar (W)",
                "battery_soc": "SOC (%)", "temperature_C": "Temp (°C)", "humidity_pct": "Humid (%) ",
                "pm25_ug_m3": "PM2.5", "ping_ms": "Ping (ms)"
            }
        ).set_index("Station"),
        use_container_width=True,
            column_config={
            "Last Update": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
            "Solar (W)": st.column_config.NumberColumn(format="%.0f"),
            "SOC (%)": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100),
            "Temp (°C)": st.column_config.NumberColumn(format="%.1f"),
            "Humid (%)": st.column_config.NumberColumn(format="%d"),
            "PM2.5": st.column_config.NumberColumn(format="%.0f µg/m³"),
            "Ping (ms)": st.column_config.NumberColumn(format="%d")
        }
    )

# -----------------------------
# Individual Station Pages
# -----------------------------
else:
    # El título ahora va debajo del header
    st.title(f"📡 Station Details: {page}")
    station_df = STATIONS[page]
    latest = station_df.iloc[-1]
    
    # --- Métricas de la Estación --- 
    st.subheader("Current Status")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="🔋 Battery SOC", value=f"{latest['battery_soc']:.0f}%")
    col2.metric(label="☀️ Solar Output", value=f"{latest['solar_W']:.0f} W")
    col3.metric(label="🌡️ Temperature", value=f"{latest['temperature_C']:.1f} °C")
    col4.metric(label="💧 Humidity", value=f"{latest['humidity_pct']:.0f}% ")
    
    col5, col6, col7, col8 = st.columns(4)
    # Calcular métricas 24h
    gen_24h_kwh = station_df['solar_Wh'].sum() / 1000
    soc_min_24h = station_df['battery_soc'].min()
    soc_max_24h = station_df['battery_soc'].max()
    pm25_avg_24h = station_df['pm25_ug_m3'].mean()
    ping_avg_24h = station_df['ping_ms'].mean()
    
    col5.metric(label="⚡ Energy Gen (24h)", value=f"{gen_24h_kwh:.2f} kWh")
    col6.metric(label="📊 SOC Range (24h)", value=f"{soc_min_24h:.0f}% - {soc_max_24h:.0f}%", help=f"Min: {soc_min_24h:.1f}%, Max: {soc_max_24h:.1f}%")
    col7.metric(label="💨 Avg PM2.5 (24h)", value=f"{pm25_avg_24h:.1f} µg/m³")
    col8.metric(label="🌐 Avg Ping (24h)", value=f"{ping_avg_24h:.0f} ms")

    st.divider()
    st.subheader("📈 Performance & Environment (Last 24 Hours)")

    # --- Gráficos de la Estación ---
    tab1, tab2 = st.tabs(["Power & Battery", "Environment & Network"])
    
    with tab1:
        # Gráfico combinado interactivo Potencia/SOC
        base_power = alt.Chart(station_df).encode(x=alt.X('timestamp:T', title='Time'))
        line_solar = base_power.mark_line(point=False, color='orange').encode(
            y=alt.Y('solar_W', axis=alt.Axis(title='Solar Power (W)', titleColor='orange')),
            tooltip=[alt.Tooltip("timestamp:T", title="Time"), alt.Tooltip("solar_W:Q", title="Solar (W)", format=".0f")]
        ).properties(title='Solar Power and Battery SOC')
        line_soc = base_power.mark_line(point=False, color='#2ecc71', strokeDash=[3,3]).encode(
            y=alt.Y('battery_soc', axis=alt.Axis(title='Battery SOC (%)', titleColor='#2ecc71')),
            tooltip=[alt.Tooltip("timestamp:T", title="Time"), alt.Tooltip("battery_soc:Q", title="SOC (%)", format=".0f")]
        )
        combined_power_chart = alt.layer(line_solar, line_soc).resolve_scale(y = 'independent').interactive()
        st.altair_chart(combined_power_chart, use_container_width=True)
        
    with tab2:
        # Gráfico combinado interactivo Ambiental/Red
        base_env = alt.Chart(station_df).encode(x=alt.X('timestamp:T', title='Time'))
        
        # Temperatura y Humedad (eje Y izquierdo)
        line_temp = base_env.mark_line(point=False, color='#e74c3c').encode(
             y=alt.Y('temperature_C', axis=alt.Axis(title='Temp (°C) / Humidity (%)', titleColor='#e74c3c')),
             tooltip=[alt.Tooltip("timestamp:T", title="Time"), alt.Tooltip("temperature_C:Q", title="Temp (°C)", format=".1f")]
        ).properties(title='Environmental Conditions & Network Quality')
        line_hum = base_env.mark_line(point=False, color='#3498db', strokeDash=[3,3]).encode(
             y=alt.Y('humidity_pct', axis=alt.Axis(title='', titleColor='#3498db')), # Eje compartido
             tooltip=[alt.Tooltip("timestamp:T", title="Time"), alt.Tooltip("humidity_pct:Q", title="Humidity (%)", format=".0f")]
        )
        env_chart = alt.layer(line_temp, line_hum)
        
        # PM2.5 y Ping (eje Y derecho)
        line_pm25 = base_env.mark_line(point=False, color='#95a5a6').encode(
             y=alt.Y('pm25_ug_m3', axis=alt.Axis(title='PM2.5 (µg/m³) / Ping (ms)', titleColor='#95a5a6')),
             tooltip=[alt.Tooltip("timestamp:T", title="Time"), alt.Tooltip("pm25_ug_m3:Q", title="PM2.5", format=".0f")]
        )
        line_ping = base_env.mark_line(point=False, color='#34495e', strokeDash=[3,3]).encode(
             y=alt.Y('ping_ms', axis=alt.Axis(title='', titleColor='#34495e')), # Eje compartido
             tooltip=[alt.Tooltip("timestamp:T", title="Time"), alt.Tooltip("ping_ms:Q", title="Ping (ms)", format=".0f")]
        )
        net_chart = alt.layer(line_pm25, line_ping)

        # Combinar ambos pares de ejes
        combined_env_chart = alt.layer(env_chart, net_chart).resolve_scale(y='independent').interactive()
        st.altair_chart(combined_env_chart, use_container_width=True)
        

    with st.expander("📄 Raw Data (Last 200 entries)"):
        # Mostrar todas las columnas relevantes
        display_cols = ["timestamp", "solar_W", "battery_soc", "temperature_C", "humidity_pct", "pm25_ug_m3", "ping_ms"]
        st.dataframe(station_df[display_cols].tail(200).set_index("timestamp"))

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption("Demo webapp – Lascar Monitoring • Generated with Streamlit ✨")

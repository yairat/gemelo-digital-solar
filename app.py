import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Configuración de la interfaz
st.set_page_config(
    page_title="Gemelo Digital Microrred Solar 3D",
    page_icon="☀️",
    layout="wide",
)

st.title("☀️ Prototipo de Gemelo Digital Fotovoltaico (3D)")
st.caption(
    "Supervisión, Mantenimiento Predictivo y Eficiencia de Microrredes Solares"
    " Urbanas — Santa Marta"
)

# 1. Carga y agrupación correcta del dataset por instante de tiempo
@st.cache_data
def cargar_y_procesar_datos():
    df_gen = pd.read_csv("Plant_1_Generation_Data.csv")
    df_sens = pd.read_csv("Plant_1_Weather_Sensor_Data.csv")

    df_gen.columns = df_gen.columns.str.strip().str.upper()
    df_sens.columns = df_sens.columns.str.strip().str.upper()

    df_gen["DATE_TIME"] = pd.to_datetime(df_gen["DATE_TIME"], dayfirst=True)
    df_sens["DATE_TIME"] = pd.to_datetime(df_sens["DATE_TIME"])

    # Promediar los inversores por cada marca de tiempo para tener 1 registro cada 15 min
    df_gen_agg = df_gen.groupby("DATE_TIME", as_index=False)[["DC_POWER", "AC_POWER"]].mean()
    df_sens_agg = df_sens.groupby("DATE_TIME", as_index=False)[["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]].mean()

    df = pd.merge_asof(
        df_gen_agg.sort_values("DATE_TIME"),
        df_sens_agg.sort_values("DATE_TIME"),
        on="DATE_TIME",
        direction="nearest"
    )

    # Normalizar DC_POWER a escala de sistema (kW) y calcular la línea base del Gemelo Digital
    # En este dataset, IRRADIATION viene normalizada (0 a ~1.2 kW/m2)
    P_STC = 14.0
    gamma = 0.004

    # Escalamos DC_POWER del dataset a nuestra planta de referencia de 14 kW para comparar 1:1
    max_dc_dataset = df["DC_POWER"].max()
    df["POTENCIA_REAL_KW"] = (df["DC_POWER"] / max_dc_dataset) * P_STC

    # Potencia esperada por el Gemelo Digital según irradiancia y temperatura del dataset
    df["POTENCIA_GEMELO_KW"] = (
        P_STC * df["IRRADIATION"] * (1 - gamma * (df["MODULE_TEMPERATURE"] - 25.0))
    ).clip(lower=0)

    # Seleccionar un día completo (96 registros de 15 min = 24 horas)
    primer_dia = df["DATE_TIME"].dt.date.unique()[1]  # Segundo día completo del dataset
    df_dia = df[df["DATE_TIME"].dt.date == primer_dia].reset_index(drop=True)
    return df_dia

df_dia = cargar_y_procesar_datos()

# 2. Sidebar - Modo de Operación (Manual vs. Simulación de Tiempo Real del Dataset)
st.sidebar.header("🕹️ Control del Gemelo Digital")
modo = st.sidebar.radio(
    "Modo de Simulación:",
    ["⏱️ Reproducción Histórica (Dataset)", "🎛️ Simulación Manual (Sliders)"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Fallas y Mantenimiento")
factor_suciedad = st.sidebar.slider("Suciedad / Polvo Costero (%)", 0, 60, 0) / 100.0
sombreado = st.sidebar.checkbox("Simular Sombreado Parcial")

P_STC = 14.0
gamma = 0.004
perdidastotales = factor_suciedad + (0.25 if sombreado else 0.0)

# Control de estado para animación Play/Pause
if "play" not in st.session_state:
    st.session_state.play = False
if "paso_tiempo" not in st.session_state:
    st.session_state.paso_tiempo = 48  # Mediodía por defecto (12:00 PM)

if modo == "⏱️ Reproducción Histórica (Dataset)":
    col_btn1, col_btn2 = st.sidebar.columns(2)
    if col_btn1.button("▶️ Play"):
        st.session_state.play = True
    if col_btn2.button("⏸️ Pausa"):
        st.session_state.play = False

    st.session_state.paso_tiempo = st.sidebar.slider(
        "Avance del Día (Cada 15 min)",
        0, len(df_dia) - 1,
        st.session_state.paso_tiempo
    )

    fila_actual = df_dia.iloc[st.session_state.paso_tiempo]
    hora_str = fila_actual["DATE_TIME"].strftime("%H:%M")
    irradiance_input = fila_actual["IRRADIATION"] * 1000.0
    temp_modulo = fila_actual["MODULE_TEMPERATURE"]

    potencia_esperada = fila_actual["POTENCIA_GEMELO_KW"]
    # Aplicamos las fallas simuladas encima de la lectura real del dataset
    potencia_real_simulada = max(0.0, fila_actual["POTENCIA_REAL_KW"] * (1 - perdidastotales))
    st.sidebar.info(f"🕒 Hora simulada: **{hora_str}** | Irradiancia real: **{irradiance_input:.0f} W/m²**")

else:
    st.session_state.play = False
    irradiance_input = st.sidebar.slider("Irradiancia Solar (W/m²)", 0.0, 1200.0, 950.0, 10.0)
    temp_amb = st.sidebar.slider("Temperatura Ambiente (°C)", 20.0, 45.0, 32.0, 0.5)
    temp_modulo = temp_amb + (irradiance_input * 0.03)
    potencia_esperada = max(0.0, P_STC * (irradiance_input / 1000.0) * (1 - gamma * (temp_modulo - 25.0)))
    potencia_real_simulada = max(0.0, potencia_esperada * (1 - perdidastotales))

desviacion = max(0.0, potencia_esperada - potencia_real_simulada)
pr_sistema = (potencia_real_simulada / potencia_esperada * 100) if potencia_esperada > 0.2 else 100.0
brillo_bombillo = min(1.0, potencia_real_simulada / P_STC)

# 3. Indicadores en Pantalla (KPIs)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Potencia Esperada (Gemelo)", f"{potencia_esperada:.2f} kW")
col2.metric("Potencia Real (Planta)", f"{potencia_real_simulada:.2f} kW")
col3.metric("Pérdida por Desviación", f"{desviacion:.2f} kW", delta_color="inverse")
col4.metric("Intensidad del Bombillo", f"{brillo_bombillo*100:.0f}%")

st.markdown("---")

# 4. MODELO 3D INTERACTIVO: PANEL SOLAR + BOMBILLO VOLUMÉTRICO
col_vis1, col_vis2 = st.columns([1.4, 1])

with col_vis1:
    st.subheader("💡 Gemelo 3D: Generación Fotovoltaica -> Carga")

    inclinacion = 0.25
    x_panel = np.linspace(0, 1.8, 20)
    y_panel = np.linspace(0, 1.0, 10)
    X, Y = np.meshgrid(x_panel, y_panel)
    Z = Y * inclinacion

    Z_color = np.full(X.shape, factor_suciedad)
    if sombreado:
        Z_color[:, :7] = 0.9

    fig_3d = go.Figure()

    # Superficie del panel solar
    fig_3d.add_trace(
        go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=Z_color,
            cmin=0, cmax=1,
            colorscale=[
                [0.0, "#104E8B"],
                [0.3, "#5C768D"],
                [0.6, "#8B6914"],
                [1.0, "#1C1C1C"],
            ],
            showscale=False,
            lighting=dict(ambient=0.7, diffuse=0.8, specular=0.5, roughness=0.4),
        )
    )

    # Celdas y Marco
    for i in range(0, 20, 3):
        fig_3d.add_trace(go.Scatter3d(x=X[:, i], y=Y[:, i], z=Z[:, i] + 0.003, mode="lines", line=dict(color="white", width=2), showlegend=False))
    for j in range(0, 10, 3):
        fig_3d.add_trace(go.Scatter3d(x=X[j, :], y=Y[j, :], z=Z[j, :] + 0.003, mode="lines", line=dict(color="white", width=2), showlegend=False))

    fig_3d.add_trace(go.Scatter3d(x=[0, 1.8, 1.8, 0, 0], y=[0, 0, 1.0, 1.0, 0], z=[0, 0, inclinacion, inclinacion, 0], mode="lines", line=dict(color="#A9A9A9", width=8), showlegend=False))

    # Cable rojo
    bx, by, bz = 2.35, 0.5, 0.35
    fig_3d.add_trace(go.Scatter3d(x=[1.8, 2.05, bx, bx], y=[0.5, 0.5, by, by], z=[inclinacion / 2, 0.02, 0.02, bz - 0.14], mode="lines", line=dict(color="#E74C3C", width=7), showlegend=False))

    # Casquillo
    fig_3d.add_trace(go.Scatter3d(x=[bx, bx], y=[by, by], z=[bz - 0.16, bz - 0.08], mode="lines", line=dict(color="#7F8C8D", width=16), showlegend=False))

    # Esfera 3D del Bombillo
    u = np.linspace(0, 2 * np.pi, 24)
    v = np.linspace(0, np.pi, 16)
    radio_bombillo = 0.14
    xs = bx + radio_bombillo * np.outer(np.cos(u), np.sin(v))
    ys = by + radio_bombillo * np.outer(np.sin(u), np.sin(v))
    zs = bz + radio_bombillo * np.outer(np.ones(np.size(u)), np.cos(v))

    if brillo_bombillo > 0.70:
        color_bombillo = "#FFF700"
    elif brillo_bombillo > 0.35:
        color_bombillo = "#F39C12"
    elif brillo_bombillo > 0.05:
        color_bombillo = "#7E5109"
    else:
        color_bombillo = "#424949"

    fig_3d.add_trace(go.Surface(x=xs, y=ys, z=zs, colorscale=[[0, color_bombillo], [1, color_bombillo]], showscale=False, opacity=0.95))

    if brillo_bombillo > 0.1:
        radio_halo = radio_bombillo + (brillo_bombillo * 0.16)
        xh = bx + radio_halo * np.outer(np.cos(u), np.sin(v))
        yh = by + radio_halo * np.outer(np.sin(u), np.sin(v))
        zh = bz + radio_halo * np.outer(np.ones(np.size(u)), np.cos(v))
        fig_3d.add_trace(go.Surface(x=xh, y=yh, z=zh, colorscale=[[0, "#FFFACD"], [1, "#FFFACD"]], showscale=False, opacity=0.15 + (brillo_bombillo * 0.25)))

    fig_3d.add_trace(go.Scatter3d(x=[bx], y=[by], z=[bz + 0.32], mode="text", text=[f"💡 {potencia_real_simulada:.1f} kW ({brillo_bombillo*100:.0f}%)"], textfont=dict(size=14, color="#E67E22"), showlegend=False))

    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-0.1, 2.8]),
            yaxis=dict(visible=False, range=[-0.2, 1.2]),
            zaxis=dict(visible=False, range=[0, 0.85]),
            camera=dict(eye=dict(x=0.0, y=-2.1, z=0.85), center=dict(x=0.05, y=0.0, z=-0.05)),
            aspectmode="manual",
            aspectratio=dict(x=2.2, y=1.0, z=0.75),
        ),
        height=390,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_3d, use_container_width=True)

with col_vis2:
    st.subheader("📊 Salud y Eficiencia del Sistema")
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pr_sistema,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Performance Ratio (PR)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2ecc71" if pr_sistema > 85 else ("#f1c40f" if pr_sistema > 70 else "#e74c3c")},
                "steps": [
                    {"range": [0, 70], "color": "#fadbd8"},
                    {"range": [70, 85], "color": "#fdebd0"},
                    {"range": [85, 100], "color": "#d4efdf"},
                ],
            },
        )
    )
    fig_gauge.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")

# 5. Módulo de Diagnóstico
st.header("🚨 Diagnóstico del Sistema y Alertas")
if potencia_esperada <= 0.2:
    st.info("🌙 **Horario Nocturno / Sin Radiación:** El sistema se encuentra en reposo esperando amanecer.")
elif perdidastotales == 0:
    st.success("✅ **Operación Óptima:** La generación real sigue fielmente la curva del Gemelo Digital.")
elif factor_suciedad > 0.15:
    st.error(f"⚠️ **Alerta de Mantenimiento Predictivo:** Brecha detectada entre el Gemelo Digital y la Planta ({factor_suciedad*100:.0f}% pérdida por suciedad). Programar limpieza.")
elif sombreado:
    st.warning("⛅ **Advertencia de Sombreado:** Desviación activa respecto a la generación teórica esperada.")

# 6. GRÁFICA COMPARATIVA DE 24 HORAS (REAL VS GEMELO DIGITAL)
st.markdown("---")
st.header("📈 Comparativo Diario (24h): Planta Real vs. Gemelo Digital")

# Aplicar la pérdida simulada a la serie para que veas cómo se separan las dos líneas
df_dia["POTENCIA_REAL_AFECTADA"] = df_dia["POTENCIA_REAL_KW"] * (1 - perdidastotales)

fig_hist = go.Figure()

# Curva 1: Gemelo Digital (Lo que DEBERÍA generar según el clima)
fig_hist.add_trace(
    go.Scatter(
        x=df_dia["DATE_TIME"],
        y=df_dia["POTENCIA_GEMELO_KW"],
        mode="lines",
        name="Gemelo Digital (Esperado)",
        line=dict(color="#F39C12", width=3, dash="dash")
    )
)

# Curva 2: Planta Real (Lo que REALMENTE genera con la suciedad/sombra actual)
fig_hist.add_trace(
    go.Scatter(
        x=df_dia["DATE_TIME"],
        y=df_dia["POTENCIA_REAL_AFECTADA"],
        mode="lines",
        name="Planta Real (Observado)",
        line=dict(color="#2980B9", width=3),
        fill="tonexty",  # Rellena el área de pérdida entre ambas curvas
        fillcolor="rgba(231, 76, 60, 0.15)"
    )
)

# Línea vertical indicando la hora actual si está en modo reproducción
if modo == "⏱️ Reproducción Histórica (Dataset)":
    hora_actual_dt = df_dia.iloc[st.session_state.paso_tiempo]["DATE_TIME"]
    fig_hist.add_vline(x=hora_actual_dt, line_width=2, line_dash="dot", line_color="#2ECC71")

fig_hist.update_layout(
    xaxis_title="Hora del Día (Ciclo Solar 24h)",
    yaxis_title="Potencia Generada (kW)",
    height=350,
    margin=dict(l=20, r=20, t=20, b=20),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_hist, use_container_width=True)

# Bucle de animación automática cuando se presiona Play
if st.session_state.play:
    time.sleep(0.4)
    st.session_state.paso_tiempo = (st.session_state.paso_tiempo + 1) % len(df_dia)
    st.rerun()
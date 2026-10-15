import io
from PIL import Image
import numpy as np
import pandas as pd
import torch
import streamlit as st

# 1. Configuración de página
st.set_page_config(
    page_title="Vision AI | Interfaz Multimodal",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inyección de CSS Avanzado + SVG Icons + Glassmorphism
st.markdown("""
<style>
    /* Estilos globales y fondo general */
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #070a12 100%);
        color: #f3f4f6;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Ocultar elementos por defecto */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Barra lateral estilo Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.75) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-right: 1px solid rgba(56, 189, 248, 0.15);
    }

    /* Tarjetas estilo Glassmorphism */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Encabezados y títulos con degradado neón */
    .gradient-text {
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Subtítulos */
    .sub-text {
        color: #94a3b8;
        font-size: 0.95rem;
    }

    /* Sliders personalizados */
    div[data-baseweb="slider"] {
        color: #38bdf8;
    }

    /* Contenedor de cámara */
    div[data-testid="stCameraInput"] {
        border: 2px dashed rgba(56, 189, 248, 0.4);
        border-radius: 16px;
        padding: 10px;
        background: rgba(15, 23, 42, 0.3);
    }

    /* Estilo de íconos SVG Inline */
    .icon-svg {
        width: 22px;
        height: 22px;
        fill: none;
        stroke: #38bdf8;
        stroke-width: 2;
        stroke-linecap: round;
        stroke-linejoin: round;
        vertical-align: middle;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# 3. Carga del modelo en caché
@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov5su.pt")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

# 4. Barra Lateral - Menú de Navegación y Configuración
with st.sidebar:
    st.markdown("""
        <h1 class='gradient-text' style='font-size: 1.6rem;'>
            <svg class="icon-svg" viewBox="0 0 24 24"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/></svg>
            Vision AI
        </h1>
        <p class='sub-text'>Navegación Multimodal</p>
    """, unsafe_allow_html=True)
    
    # Navegación entre funciones
    opcion_menu = st.radio(
        "Selecciona el Módulo:",
        ["Detección YOLOv5", "Teachable Machine (Gestos)"],
        index=0
    )
    
    st.markdown("---")

    # Si estamos en el módulo YOLO, mostramos la configuración
    if opcion_menu == "Detección YOLOv5":
        st.markdown("""
            <h3 style='color: #38bdf8; font-size: 1rem; display: flex; align-items: center; gap: 8px;'>
                <svg class="icon-svg" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="M20 12h2"/><path d="M2 12h2"/></svg>
                Parámetros de Detección
            </h3>
        """, unsafe_allow_html=True)
        conf_threshold = st.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)
        iou_threshold  = st.slider("Umbral IoU", 0.0, 1.0, 0.45, 0.01)
        max_det        = st.number_input("Detecciones máximas", 10, 2000, 1000, 10)

# 5. MÓDULO 1: DETECCIÓN CON YOLOV5
if opcion_menu == "Detección YOLOv5":
    # Encabezado Principal
    st.markdown("""
        <div class="glass-card">
            <h1 class="gradient-text" style="margin:0;">
                <svg class="icon-svg" style="width: 32px; height: 32px;" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
                Detección de Objetos en Tiempo Real
            </h1>
            <p class="sub-text" style="margin-top: 8px;">
                Captura un fotograma desde tu cámara web para procesarlo con la arquitectura de visión artificial <b>YOLOv5</b>.
            </p>
        </div>
    """, unsafe_allow_html=True)

    with st.spinner("Cargando arquitectura YOLOv5..."):
        model = load_model()

    if model:
        picture = st.camera_input("Capturar imagen desde la cámara", key="camera")

        if picture:
            bytes_data = picture.getvalue()

            # Decodificar imagen para YOLO
            pil_img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
            np_img  = np.array(pil_img)[..., ::-1]  # RGB → BGR

            with st.spinner("Procesando fotograma con IA..."):
                try:
                    results = model(
                        np_img,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        max_det=int(max_det)
                    )
                except Exception as e:
                    st.error(f"Error durante la detección: {str(e)}")
                    st.stop()

            result        = results[0]
            boxes         = result.boxes
            annotated     = result.plot()
            annotated_rgb = annotated[:, :, ::-1]  # BGR → RGB

            # Layout en 2 Columnas
            col1, col2 = st.columns([1.2, 1])

            with col1:
                st.markdown("""
                    <div class="glass-card">
                        <h3 style="color: #38bdf8; margin-top:0; display: flex; align-items: center; gap: 8px;">
                            <svg class="icon-svg" viewBox="0 0 24 24"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
                            Resultado Visual
                        </h3>
                """, unsafe_allow_html=True)
                st.image(annotated_rgb, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown("""
                    <div class="glass-card">
                        <h3 style="color: #38bdf8; margin-top:0; display: flex; align-items: center; gap: 8px;">
                            <svg class="icon-svg" viewBox="0 0 24 24"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
                            Métricas de Detección
                        </h3>
                """, unsafe_allow_html=True)
                
                if boxes is not None and len(boxes) > 0:
                    label_names    = model.names
                    category_count = {}
                    category_conf  = {}

                    for box in boxes:
                        cat

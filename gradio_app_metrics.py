import gradio as gr
from PIL import Image
import torch
from io import BytesIO
import time
import tempfile
import os
import trimesh
import numpy as np
import logging
import traceback

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen.pipelines import Hunyuan3DDiTFlowMatchingPipeline

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar el pipeline al iniciar
try:
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2',
        subfolder='hunyuan3d-dit-v2-0',
        variant='fp16'
    )
    logger.info("Modelo 3D cargado correctamente")
except Exception as e:
    logger.error(f"Error al cargar el modelo 3D: {str(e)}")
    pipeline = None

def calculate_basic_metrics(mesh):
    metrics = {
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
        "area": float(mesh.area) if hasattr(mesh, 'area') else 0,
        "bounding_box_dimensions": (
            mesh.bounding_box.primitive.extents.tolist() 
            if hasattr(mesh, 'bounding_box') and mesh.bounding_box is not None 
            else "No disponible"
        ),
        "is_watertight": bool(mesh.is_watertight) if hasattr(mesh, 'is_watertight') else False,
        "volume": float(mesh.volume) if getattr(mesh, 'is_watertight', False) else "No calculable"
    }
    return metrics

def calculate_advanced_metrics(mesh):
    metrics = {}
    try:
        if hasattr(mesh, 'area') and mesh.area > 0:
            metrics["vertex_density"] = len(mesh.vertices) / mesh.area
        else:
            metrics["vertex_density"] = 0

        edges = set()
        for face in mesh.faces[:1000]:  # limitar a 1000 caras para evitar bloqueos
            for i in range(3):
                edges.add(tuple(sorted((face[i], face[(i+1)%3]))))

        metrics["edge_count"] = len(edges)
        metrics["edge_to_vertex_ratio"] = len(edges) / len(mesh.vertices) if len(mesh.vertices) > 0 else 0
    except Exception as e:
        logger.warning(f"Error calculando métricas avanzadas: {str(e)}")
    return metrics

def process_image_and_generate_3d(input_image: Image.Image):
    if pipeline is None:
        return None, None, "❌ Error: Modelo 3D no disponible.", "⏱️ Tiempo: N/A"

    try:
        # Convertir imagen a RGBA y quitar fondo
        image = input_image.convert("RGBA")
        rembg = BackgroundRemover()
        image = rembg(image)

        # Generar el modelo 3D
        start_time = time.time()
        mesh = pipeline(
            image=image,
            num_inference_steps=50,
            octree_resolution=380,
            num_chunks=20000,
            generator=torch.manual_seed(12345),
            output_type='trimesh'
        )[0]
        elapsed = time.time() - start_time
        logger.info(f"Modelo generado en {elapsed:.2f} segundos")

        # Guardar modelo .glb
        with tempfile.NamedTemporaryFile(delete=False, suffix=".glb") as tmp_file:
            mesh.export(tmp_file.name)
            glb_path = tmp_file.name

        # Calcular métricas
        basic = calculate_basic_metrics(mesh)
        advanced = calculate_advanced_metrics(mesh)
        metrics = {**basic, **advanced, "generation_time": f"{elapsed:.2f} s"}

        # Formatear tiempo de generación para mostrar
        time_display = f"⏱️ Tiempo de generación: {elapsed:.2f} segundos"
        if elapsed > 60:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            time_display = f"⏱️ Tiempo de generación: {minutes}m {seconds:.2f}s"

        return image, glb_path, metrics, time_display

    except Exception as e:
        logger.error(f"Error procesando imagen: {str(e)}\n{traceback.format_exc()}")
        return None, None, f"❌ Error inesperado: {str(e)}", "⏱️ Tiempo: Error"

# Interfaz Gradio
with gr.Blocks(title="Generador 3D con Hunyuan3D") as demo:
    gr.Markdown("## 🧠 Generador de modelos 3D a partir de imágenes con métricas")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="📤 Sube tu imagen")
            generate_btn = gr.Button("🚀 Generar Modelo 3D", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(type="pil", label="🖼 Imagen procesada")
            
            # Componente dedicado para mostrar el tiempo de generación
            time_display = gr.Textbox(
                label="⏱️ Tiempo de Generación", 
                value="Presiona 'Generar Modelo 3D' para comenzar...",
                interactive=False,
                show_copy_button=True
            )
            
            glb_download = gr.File(label="📦 Descargar modelo .glb")
            metrics_output = gr.JSON(label="📊 Métricas del Modelo")

    generate_btn.click(
        fn=process_image_and_generate_3d,
        inputs=[input_image],
        outputs=[output_image, glb_download, metrics_output, time_display]
    )

demo.launch()
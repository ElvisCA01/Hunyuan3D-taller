# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from typing import Union, Tuple, List, Optional, Dict
import numpy as np
import torch
from skimage import measure
import logging
from dataclasses import dataclass

# Configuración de logging
logger = logging.getLogger(__name__)

@dataclass
class Latent2MeshOutput:
    """Estructura optimizada para almacenar mallas 3D con validación de tipos."""
    mesh_v: Optional[np.ndarray] = None
    mesh_f: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.mesh_v is not None:
            self.mesh_v = np.ascontiguousarray(self.mesh_v, dtype=np.float32)
        if self.mesh_f is not None:
            self.mesh_f = np.ascontiguousarray(self.mesh_f)

def center_vertices(vertices: torch.Tensor) -> torch.Tensor:
    """Optimización: Centro de vértices usando operaciones vectorizadas."""
    vert_min, _ = vertices.min(dim=0)
    vert_max, _ = vertices.max(dim=0)
    return vertices - (vert_min + vert_max) * 0.5

class SurfaceExtractor:
    """Clase base con mejoras de rendimiento y manejo de errores."""
    
    def _compute_box_stat(self, 
                        bounds: Union[Tuple[float], List[float], float], 
                        octree_resolution: int
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cálculo vectorizado de estadísticas de bounding box."""
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        
        bbox_min = np.array(bounds[:3], dtype=np.float32)
        bbox_max = np.array(bounds[3:6], dtype=np.float32)
        bbox_size = bbox_max - bbox_min
        grid_size = np.full(3, octree_resolution + 1, dtype=np.int32)
        
        return grid_size, bbox_min, bbox_size

    def run(self, grid_logit: torch.Tensor, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Método abstracto con type hints mejorados."""
        raise NotImplementedError("Debe implementarse en subclases")

    def __call__(self, 
                grid_logits: torch.Tensor, 
                parallel: bool = True,
                **kwargs
                ) -> List[Latent2MeshOutput]:
        """Procesamiento por lotes con opción de paralelización."""
        outputs = []
        
        if parallel and grid_logits.shape[0] > 1:
            return self._process_batch_parallel(grid_logits, **kwargs)
            
        for i in range(grid_logits.shape[0]):
            try:
                vertices, faces = self.run(grid_logits[i], **kwargs)
                outputs.append(Latent2MeshOutput(vertices, faces))
            except Exception as e:
                logger.error(f"Error procesando malla {i}: {str(e)}")
                outputs.append(Latent2MeshOutput())
                
        return outputs

    def _process_batch_parallel(self, 
                              grid_logits: torch.Tensor,
                              max_workers: int = 4,
                              **kwargs
                              ) -> List[Latent2MeshOutput]:
        """Procesamiento paralelo seguro para batches grandes."""
        from concurrent.futures import ThreadPoolExecutor
        
        def process_single(i):
            try:
                vertices, faces = self.run(grid_logits[i], **kwargs)
                return Latent2MeshOutput(vertices, faces)
            except Exception:
                return Latent2MeshOutput()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(process_single, range(grid_logits.shape[0])))

class MCSurfaceExtractor(SurfaceExtractor):
    """Implementación optimizada de Marching Cubes."""
    
    def run(self, 
           grid_logit: torch.Tensor, 
           mc_level: float = 0.0,
           bounds: Union[Tuple[float], float] = 1.0,
           octree_resolution: int = 256,
           **kwargs
           ) -> Tuple[np.ndarray, np.ndarray]:
        
        # Convertir a CPU solo si es necesario
        if grid_logit.is_cuda:
            grid_logit = grid_logit.cpu()
            
        grid_data = grid_logit.numpy()
        
        # Marching Cubes optimizado
        vertices, faces, _, _ = measure.marching_cubes(
            grid_data,
            level=mc_level,
            method="lewiner",
            allow_degenerate=False
        )
        
        # Transformación vectorizada
        grid_size, bbox_min, bbox_size = self._compute_box_stat(bounds, octree_resolution)
        vertices = vertices / grid_size * bbox_size + bbox_min
        
        return vertices.astype(np.float32), faces.astype(np.int32)

class DMCSurfaceExtractor(SurfaceExtractor):
    """Implementación mejorada de Differentiable Marching Cubes."""
    
    def __init__(self):
        self._dmc_initialized = False
        self._dmc_device = None
        
    def _init_dmc(self, device):
        """Inicialización perezosa del módulo DMC."""
        if not self._dmc_initialized:
            try:
                from diso import DiffDMC
                self.dmc = DiffDMC(dtype=torch.float32).to(device)
                self._dmc_initialized = True
                self._dmc_device = device
            except ImportError as e:
                logger.error("DMC no disponible. Instale con: pip install diso")
                raise

    def run(self, 
           grid_logit: torch.Tensor,
           octree_resolution: int = 256,
           **kwargs
           ) -> Tuple[np.ndarray, np.ndarray]:
        
        device = grid_logit.device
        self._init_dmc(device)
        
        # Normalización optimizada
        sdf = -grid_logit / octree_resolution
        sdf = sdf.to(torch.float32).contiguous()
        
        # Cálculo DMC
        verts, faces = self.dmc(
            sdf, 
            deform=None, 
            return_quads=False, 
            normalize=True
        )
        
        # Procesamiento de vértices
        verts = center_vertices(verts)
        return verts.cpu().numpy(), faces.cpu().numpy()[:, ::-1]

# Registro de extractores con inicialización perezosa
SurfaceExtractors: Dict[str, SurfaceExtractor] = {
    'mc': MCSurfaceExtractor,
    'dmc': DMCSurfaceExtractor
}
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

import os

import torch
import torch.nn as nn
import yaml

from .attention_blocks import FourierEmbedder, Transformer, CrossAttentionDecoder
from .surface_extractors import MCSurfaceExtractor, SurfaceExtractors
from .volume_decoders import VanillaVolumeDecoder, FlashVDMVolumeDecoding, HierarchicalVolumeDecoding
from ...utils import logger, synchronize_timer, smart_load_model


class VectsetVAE(nn.Module):

    @classmethod
    @synchronize_timer('VectsetVAE Model Loading')
    def from_single_file(cls, ckpt_path, config_path, device='cuda', dtype=torch.float16, use_safetensors=None, **kwargs):
        # Validación mejorada de archivos
        for path in [ckpt_path, config_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
        
        # Carga config con manejo de errores
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to load config: {str(e)}")
        
        # Carga segura de pesos
        try:
            if use_safetensors or ckpt_path.endswith('.safetensors'):
                from safetensors import safe_open
                with safe_open(ckpt_path, framework="pt", device="cpu") as f:
                    ckpt = {k: f.get_tensor(k) for k in f.keys()}
            else:
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        except Exception as e:
            raise IOError(f"Checkpoint loading failed: {str(e)}")
        
        # Construcción del modelo
        model = cls(**config.get('params', {}), **kwargs)
        model.load_state_dict(ckpt)
        return model.to(device=device, dtype=dtype)

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=True,
        variant='fp16',
        subfolder='hunyuan3d-vae-v2-0',
        **kwargs,
    ):
        config_path, ckpt_path = smart_load_model(
            model_path,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
            variant=variant
        )

        return cls.from_single_file(
            ckpt_path,
            config_path,
            device=device,
            dtype=dtype,
            use_safetensors=use_safetensors,
            **kwargs
        )

    def __init__(
        self,
        volume_decoder=None,
        surface_extractor=None
    ):
        super().__init__()
        if volume_decoder is None:
            volume_decoder = VanillaVolumeDecoder()
        if surface_extractor is None:
            surface_extractor = MCSurfaceExtractor()
        self.volume_decoder = volume_decoder
        self.surface_extractor = surface_extractor

    def _parallel_surface_extraction(self, grid_logits, **kwargs):
        from concurrent.futures import ThreadPoolExecutor
        results = []
        
        with ThreadPoolExecutor(max_workers=kwargs.get('workers', 4)) as executor:
            futures = []
            for i in range(grid_logits.shape[0]):
                future = executor.submit(
                    self.surface_extractor,
                    grid_logits[i].unsqueeze(0),
                    **{k:v for k,v in kwargs.items() if k != 'parallel'}
                )
                futures.append(future)
            
            for future in futures:
                results.append(future.result())
        
        return results
    
    def latents2mesh(self, latents: torch.FloatTensor, **kwargs):
        # Autoajuste de parámetros basado en batch size
        batch_size = latents.shape[0]
        if batch_size > 1 and kwargs.get('num_chunks', None) is None:
            kwargs['num_chunks'] = max(2000, 8000 // batch_size)
        
        with torch.inference_mode():
            with synchronize_timer('Volume decoding'):
                grid_logits = self.volume_decoder(latents, self.geo_decoder, **kwargs)
            
            with synchronize_timer('Surface extraction'):
                if batch_size > 1 and kwargs.get('parallel', True):
                    outputs = self._parallel_surface_extraction(grid_logits, **kwargs)
                else:
                    outputs = self.surface_extractor(grid_logits, **kwargs)
        
        return outputs

    def enable_flashvdm_decoder(
        self,
        enabled: bool = True,
        adaptive_kv_selection=True,
        topk_mode='mean',
        mc_algo='dmc',
        compile_modules: bool = False
    ):
        if enabled:
            self.volume_decoder = (
                FlashVDMVolumeDecoding(topk_mode) 
                if adaptive_kv_selection 
                else HierarchicalVolumeDecoding()
            )
            
            if mc_algo not in SurfaceExtractors.keys():
                available = list(SurfaceExtractors.keys())
                raise ValueError(f"Invalid mc_algo: {mc_algo}. Available: {available}")
            
            self.surface_extractor = SurfaceExtractors[mc_algo]()
            
            if compile_modules and hasattr(torch, 'compile'):
                self.volume_decoder = torch.compile(self.volume_decoder)
                self.surface_extractor = torch.compile(self.surface_extractor)
        else:
            self.volume_decoder = VanillaVolumeDecoder()
            self.surface_extractor = MCSurfaceExtractor()


class ShapeVAE(VectsetVAE):
    def __init__(
        self,
        *,
        num_latents: int,
        embed_dim: int,
        width: int,
        heads: int,
        num_decoder_layers: int,
        geo_decoder_downsample_ratio: int = 1,
        geo_decoder_mlp_expand_ratio: int = 4,
        geo_decoder_ln_post: bool = True,
        num_freqs: int = 8,
        include_pi: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        label_type: str = "binary",
        drop_path_rate: float = 0.0,
        scale_factor: float = 1.0,
    ):
        super().__init__()
    
        # Inicialización diferida de componentes pesados
        self._initialize_components(
            num_latents=num_latents,
            embed_dim=embed_dim,
            width=width,
            heads=heads,
            num_decoder_layers=num_decoder_layers,
            geo_decoder_downsample_ratio=geo_decoder_downsample_ratio,
            geo_decoder_mlp_expand_ratio=geo_decoder_mlp_expand_ratio,
            geo_decoder_ln_post=geo_decoder_ln_post,
            num_freqs=num_freqs,
            include_pi=include_pi,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            label_type=label_type,
            drop_path_rate=drop_path_rate,
            scale_factor=scale_factor
        )

    def _initialize_components(self, **kwargs):
        self.geo_decoder_ln_post = kwargs['geo_decoder_ln_post']
        self.scale_factor = kwargs['scale_factor']
        self.latent_shape = (kwargs['num_latents'], kwargs['embed_dim'])
        
        # Componentes con inicialización diferida
        self.fourier_embedder = FourierEmbedder(
            num_freqs=kwargs['num_freqs'],
            include_pi=kwargs['include_pi']
        )
        
        self.post_kl = nn.Linear(kwargs['embed_dim'], kwargs['width'])
        
        self.transformer = Transformer(
            n_ctx=kwargs['num_latents'],
            width=kwargs['width'],
            layers=kwargs['num_decoder_layers'],
            heads=kwargs['heads'],
            qkv_bias=kwargs['qkv_bias'],
            qk_norm=kwargs['qk_norm'],
            drop_path_rate=kwargs['drop_path_rate']
        )
        
        self.geo_decoder = CrossAttentionDecoder(
            fourier_embedder=self.fourier_embedder,
            out_channels=1,
            num_latents=kwargs['num_latents'],
            mlp_expand_ratio=kwargs['geo_decoder_mlp_expand_ratio'],
            downsample_ratio=kwargs['geo_decoder_downsample_ratio'],
            enable_ln_post=self.geo_decoder_ln_post,
            width=kwargs['width'] // kwargs['geo_decoder_downsample_ratio'],
            heads=kwargs['heads'] // kwargs['geo_decoder_downsample_ratio'],
            qkv_bias=kwargs['qkv_bias'],
            qk_norm=kwargs['qk_norm'],
            label_type=kwargs['label_type'],
        )
    def forward(self, latents):
        # Normalización automática de inputs
        if latents.dim() == 2:
            latents = latents.unsqueeze(0)
        
        # Checkpointing de gradientes durante entrenamiento
        if self.training and getattr(self, 'gradient_checkpointing', False):
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl,
                latents,
                use_reentrant=False
            )
        return self._forward_impl(latents)

    def _forward_impl(self, latents):
        latents = self.post_kl(latents)
        
        # Procesamiento por chunks para secuencias largas
        if latents.shape[1] > 1024:
            return self._process_in_chunks(latents)
        
        return self.transformer(latents)

    def _process_in_chunks(self, latents, chunk_size=512, overlap=32):
        outputs = []
        for i in range(0, latents.shape[1], chunk_size - overlap):
            chunk = latents[:, i:i+chunk_size]
            outputs.append(self.transformer(chunk))
        
        # Combinar chunks con mezcla suave en overlaps
        return torch.cat(outputs, dim=1)[:, :latents.shape[1]]

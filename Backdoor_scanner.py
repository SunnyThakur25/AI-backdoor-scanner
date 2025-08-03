#!/usr/bin/env python3
"""
BackdoorScanner - Advanced AI Model Backdoor Detection Tool
Version: 2.1
Author: Red Team Operations
License: MIT
Description: Comprehensive tool for detecting potential backdoors in AI/ML models using multiple detection techniques.
"""

import os
import sys
import json
import argparse
import numpy as np
import pickle
import hashlib
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
import logging
from enum import Enum, auto
from dataclasses import dataclass
import platform
import time
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('BackdoorScanner')

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
DEFAULT_DETECTION_METHODS = [
    'activation_clustering',
    'neuron_inspection',
    'adversarial_analysis',
    'spectral_analysis',
    'input_output_analysis',
    'weight_analysis'
]

MAX_REPORT_EVIDENCE = 10  # Limit evidence items in report
MIN_SAMPLES_FOR_CLUSTERING = 5  # Minimum samples for clustering analysis

class ModelType(Enum):
    TEXT = auto()
    VISION = auto()
    MULTIMODAL = auto()
    UNKNOWN = auto()

    def __str__(self):
        return self.name.lower()

class ModelFramework(Enum):
    PYTORCH = auto()
    TENSORFLOW = auto()
    ONNX = auto()
    UNKNOWN = auto()

    def __str__(self):
        return self.name.lower()

class ConfidenceLevel(Enum):
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()

    def __str__(self):
        return self.name.lower()

@dataclass
class DetectionResult:
    method: str
    description: str
    score: float = 0.0
    findings: List[str] = None
    evidence: List[Dict] = None
    execution_time: float = 0.0
    error: Optional[str] = None

    def __post_init__(self):
        if self.findings is None:
            self.findings = []
        if self.evidence is None:
            self.evidence = []

# Framework-specific imports with better error handling
def import_required_libraries():
    """Dynamically import required libraries with proper error handling"""
    global HAS_PYTORCH, HAS_TENSORFLOW, HAS_TRANSFORMERS, HAS_MODELSCOPE, HAS_ONNX
    global HAS_SKLEARN, HAS_VISUALIZATION, HAS_DATASETS, HAS_WORDCLOUD

    HAS_PYTORCH = HAS_TENSORFLOW = HAS_TRANSFORMERS = False
    HAS_MODELSCOPE = HAS_ONNX = HAS_SKLEARN = HAS_VISUALIZATION = False
    HAS_DATASETS = HAS_WORDCLOUD = False

    try:
        import torch
        import torch.nn as nn
        HAS_PYTORCH = True
    except ImportError:
        logger.warning("PyTorch not available. PyTorch models won't be supported.")

    try:
        import tensorflow as tf
        from tensorflow import keras
        HAS_TENSORFLOW = True
    except ImportError:
        logger.warning("TensorFlow not available. TensorFlow models won't be supported.")

    try:
        from transformers import (
            AutoModel, AutoTokenizer,
            AutoModelForSequenceClassification,
            AutoModelForImageClassification
        )
        HAS_TRANSFORMERS = True
    except ImportError:
        logger.warning("Transformers not available. HuggingFace models won't be supported.")

    try:
        import modelscope
        from modelscope.models import Model
        HAS_MODELSCOPE = True
    except ImportError:
        logger.warning("ModelScope not available. ModelScope models won't be supported.")

    try:
        import onnx
        import onnxruntime as ort
        HAS_ONNX = True
    except ImportError:
        logger.warning("ONNX not available. ONNX models won't be supported.")

    try:
        from sklearn.decomposition import PCA, NMF
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.ensemble import IsolationForest
        from sklearn.svm import OneClassSVM
        from sklearn.metrics import pairwise_distances
        from scipy import stats
        from scipy.spatial.distance import cosine
        from scipy.stats import entropy, ks_2samp
        HAS_SKLEARN = True
    except ImportError:
        logger.warning("scikit-learn not available. Some detection methods won't work.")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.colors import LinearSegmentedColormap
        HAS_VISUALIZATION = True
    except ImportError:
        logger.warning("Visualization libraries not available. Report visualization won't work.")

    try:
        from datasets import load_dataset
        HAS_DATASETS = True
    except ImportError:
        logger.warning("HuggingFace datasets not available. Some dataset loading features won't work.")

    try:
        from wordcloud import WordCloud
        HAS_WORDCLOUD = True
    except ImportError:
        logger.warning("WordCloud not available. Visualization word clouds won't work.")

# Import libraries at module level
import_required_libraries()

class ModelLoader:
    """Handles model loading from different sources and frameworks"""
    
    @staticmethod
    def load_model(model_path: str, model_source: str, model_type: str, framework: str):
        """Factory method to load model based on source and framework"""
        model_source = model_source.lower()
        model_type = model_type.lower()
        framework = framework.lower()
        
        if model_source == "huggingface" and HAS_TRANSFORMERS:
            return ModelLoader._load_huggingface_model(model_path, model_type)
        elif model_source == "modelscope" and HAS_MODELSCOPE:
            return ModelLoader._load_modelscope_model(model_path, model_type)
        else:
            return ModelLoader._load_local_model(model_path, framework)
    
    @staticmethod
    def _load_huggingface_model(model_path: str, model_type: str):
        """Load model from Hugging Face Hub"""
        try:
            tokenizer = None
            model = None
            model_info = {}
            
            if model_type in ["auto", "text"]:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModelForSequenceClassification.from_pretrained(model_path)
                    framework = "pytorch" if isinstance(model, torch.nn.Module) else "tensorflow"
                    model_info = {
                        'model_class': model.__class__.__name__,
                        'num_labels': getattr(model, 'num_labels', None),
                        'config': model.config.to_dict() if hasattr(model, 'config') else {}
                    }
                    return model, tokenizer, model_info, framework, ModelType.TEXT
                except Exception as e:
                    logger.debug(f"Failed to load as text model: {str(e)}")
            
            if model_type in ["auto", "vision"]:
                try:
                    model = AutoModelForImageClassification.from_pretrained(model_path)
                    framework = "pytorch" if isinstance(model, torch.nn.Module) else "tensorflow"
                    model_info = {
                        'model_class': model.__class__.__name__,
                        'num_labels': getattr(model, 'num_labels', None),
                        'config': model.config.to_dict() if hasattr(model, 'config') else {}
                    }
                    return model, None, model_info, framework, ModelType.VISION
                except Exception as e:
                    logger.debug(f"Failed to load as vision model: {str(e)}")
            
            # Fallback to generic model loading
            model = AutoModel.from_pretrained(model_path)
            framework = "pytorch" if isinstance(model, torch.nn.Module) else "tensorflow"
            model_info = {
                'model_class': model.__class__.__name__,
                'config': model.config.to_dict() if hasattr(model, 'config') else {}
            }
            return model, None, model_info, framework, ModelType.UNKNOWN
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {str(e)}")
            raise ValueError(f"Failed to load HuggingFace model: {str(e)}")
    
    @staticmethod
    def _load_modelscope_model(model_path: str, model_type: str):
        """Load model from ModelScope"""
        try:
            if model_type in ["auto", "text"]:
                try:
                    model = Model.from_pretrained(model_path)
                    tokenizer = modelscope.AutoTokenizer.from_pretrained(model_path)
                    framework = "pytorch" if isinstance(model, torch.nn.Module) else "tensorflow"
                    model_info = {
                        'model_class': model.__class__.__name__,
                        'model_id': getattr(model, 'model_id', None),
                        'task': getattr(model, 'task', None)
                    }
                    return model, tokenizer, model_info, framework, ModelType.TEXT
                except Exception as e:
                    logger.debug(f"Failed to load as text model: {str(e)}")
            
            # Fallback to generic model loading
            model = Model.from_pretrained(model_path)
            framework = "pytorch" if isinstance(model, torch.nn.Module) else "tensorflow"
            model_info = {
                'model_class': model.__class__.__name__,
                'model_id': getattr(model, 'model_id', None),
                'task': getattr(model, 'task', None)
            }
            return model, None, model_info, framework, ModelType.UNKNOWN
            
        except Exception as e:
            logger.error(f"Failed to load ModelScope model: {str(e)}")
            raise ValueError(f"Failed to load ModelScope model: {str(e)}")
    
    @staticmethod
    def _load_local_model(model_path: str, framework: str):
        """Load local model file"""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            ext = model_path.suffix.lower()
            
            if ext in ['.pt', '.pth'] and HAS_PYTORCH:
                model = torch.load(model_path, map_location=torch.device('cpu'))
                framework = "pytorch"
                model_info = {
                    'model_class': model.__class__.__name__ if isinstance(model, torch.nn.Module) else "state_dict",
                    'state_dict_keys': list(model.state_dict().keys()) if isinstance(model, torch.nn.Module) else list(model.keys())
                }
                return model, None, model_info, framework, ModelType.UNKNOWN
            
            elif ext in ['.h5', '.keras'] and HAS_TENSORFLOW:
                model = keras.models.load_model(model_path)
                framework = "tensorflow"
                model_info = {
                    'model_class': model.__class__.__name__,
                    'input_shape': model.input_shape,
                    'output_shape': model.output_shape,
                    'layers': [layer.name for layer in model.layers]
                }
                return model, None, model_info, framework, ModelType.UNKNOWN
            
            elif ext == '.onnx' and HAS_ONNX:
                model = onnx.load(model_path)
                onnx.checker.check_model(model)
                framework = "onnx"
                model_info = {
                    'graph_name': model.graph.name,
                    'input_info': [(inp.name, str(inp.type)) for inp in model.graph.input],
                    'output_info': [(out.name, str(out.type)) for out in model.graph.output],
                    'nodes': len(model.graph.node)
                }
                return model, None, model_info, framework, ModelType.UNKNOWN
            
            else:
                raise ValueError(f"Unsupported model format: {ext}")
                
        except Exception as e:
            logger.error(f"Failed to load local model: {str(e)}")
            raise ValueError(f"Failed to load local model: {str(e)}")

class BackdoorDetector:
    """Main class for backdoor detection in AI models"""
    
    def __init__(self, model_path: str, model_source: str = "local", 
                 model_type: str = "auto", framework: str = "auto", 
                 verbose: bool = False):
        """
        Initialize the backdoor detector
        
        Args:
            model_path: Path to model or model identifier
            model_source: Source of model (huggingface, modelscope, local)
            model_type: Type of model (text, vision, multimodal, auto)
            framework: Framework of model (pytorch, tensorflow, onnx, auto)
            verbose: Enable verbose logging
        """
        self.model_path = model_path
        self.model_source = model_source.lower()
        self.model_type = model_type.lower()
        self.framework = framework.lower()
        self.verbose = verbose
        
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"Initializing BackdoorDetector for model: {model_path}")
        logger.debug(f"Parameters: source={model_source}, type={model_type}, framework={framework}")
        
        # Load the model
        self.model, self.tokenizer, self.model_info, self.framework, self.model_type = \
            ModelLoader.load_model(model_path, model_source, model_type, framework)
        
        # Initialize detection results
        self.detection_results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'model_source': self.model_source,
            'model_type': str(self.model_type),
            'framework': self.framework,
            'methods_used': [],
            'findings': [],
            'backdoor_score': 0.0,
            'confidence': 'low',
            'system_info': self._get_system_info(),
            'model_info': self.model_info
        }
        
        logger.info("Model loaded successfully")
    
    def _get_system_info(self) -> Dict:
        """Get system information for reproducibility"""
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu': platform.processor(),
            'system_time': datetime.now().isoformat()
        }
    
    def _prepare_data(self, data_path: str) -> Any:
        """Prepare clean dataset for comparison"""
        if not data_path:
            logger.debug("No data path provided, generating synthetic data")
            return self._generate_synthetic_data()
        
        try:
            if self.model_source == "huggingface" and HAS_TRANSFORMERS and HAS_DATASETS:
                try:
                    dataset = load_dataset(data_path)
                    logger.debug(f"Loaded dataset from HuggingFace: {data_path}")
                    return dataset
                except Exception as e:
                    logger.debug(f"Failed to load HuggingFace dataset: {str(e)}")
            
            # Try loading as numpy array
            try:
                data = np.load(data_path, allow_pickle=True)
                logger.debug(f"Loaded data from numpy file: {data_path}")
                return data
            except Exception as e:
                logger.debug(f"Failed to load numpy data: {str(e)}")
            
            # Try loading as image files
            try:
                if HAS_VISUALIZATION:
                    from PIL import Image
                    import glob
                    
                    if os.path.isdir(data_path):
                        image_files = glob.glob(os.path.join(data_path, "*.jpg")) + \
                                    glob.glob(os.path.join(data_path, "*.png"))
                        images = [np.array(Image.open(f)) for f in image_files[:10]]  # Load first 10 images
                        if images:
                            return np.stack(images)
            except Exception as e:
                logger.debug(f"Failed to load image data: {str(e)}")
            
            logger.warning("Could not load dataset, generating synthetic data")
            return self._generate_synthetic_data()
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> Any:
        """Generate synthetic data for analysis"""
        try:
            if self.model_type == ModelType.TEXT and self.tokenizer:
                synthetic_texts = [
                    "This is a sample text for analysis.",
                    "Another example sentence for testing.",
                    "The model will process this input.",
                    "Synthetic data for backdoor detection.",
                    "Testing the model with generated text."
                ]
                inputs = self.tokenizer(synthetic_texts, return_tensors="pt", padding=True, truncation=True)
                logger.debug("Generated synthetic text data")
                return inputs
            
            elif self.model_type == ModelType.VISION:
                synthetic_images = np.random.rand(5, 3, 224, 224).astype(np.float32)
                logger.debug("Generated synthetic image data")
                return torch.tensor(synthetic_images)
            
            else:
                synthetic_data = np.random.rand(10, 100).astype(np.float32)
                logger.debug("Generated generic synthetic data")
                return synthetic_data
                
        except Exception as e:
            logger.error(f"Error generating synthetic data: {str(e)}")
            return np.random.rand(5, 100).astype(np.float32)
    
    def _extract_activations(self, data: Any) -> Dict:
        """Extract activations from intermediate layers"""
        activations = {}
        
        try:
            if self.framework == "pytorch" and isinstance(self.model, torch.nn.Module):
                hooks = []
                
                def get_activation(name):
                    def hook(model, input, output):
                        activations[name] = output.detach()
                    return hook
                
                # Register hooks for selected layers
                for name, layer in self.model.named_modules():
                    if len(list(layer.children())) == 0:  # Leaf module
                        if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d)):
                            hooks.append(layer.register_forward_hook(get_activation(name)))
                
                # Run inference
                with torch.no_grad():
                    if isinstance(data, dict) and 'input_ids' in data:
                        outputs = self.model(**data)
                    elif isinstance(data, torch.Tensor):
                        outputs = self.model(data)
                    else:
                        data_tensor = torch.tensor(data, dtype=torch.float32)
                        outputs = self.model(data_tensor)
                
                # Remove hooks
                for hook in hooks:
                    hook.remove()
                    
            elif self.framework == "tensorflow" and hasattr(self.model, 'layers'):
                layer_outputs = [layer.output for layer in self.model.layers 
                               if isinstance(layer, (tf.keras.layers.Dense, 
                                                   tf.keras.layers.Conv2D, 
                                                   tf.keras.layers.Conv1D))]
                if layer_outputs:
                    activation_model = tf.keras.Model(inputs=self.model.input, outputs=layer_outputs)
                    
                    if isinstance(data, dict) and 'input_ids' in data:
                        activations_list = activation_model({
                            'input_ids': data['input_ids'].numpy(),
                            'attention_mask': data.get('attention_mask', torch.ones_like(data['input_ids'])).numpy()
                        })
                    elif isinstance(data, torch.Tensor):
                        activations_list = activation_model(data.numpy())
                    else:
                        activations_list = activation_model(data)
                    
                    for i, layer in enumerate([l for l in self.model.layers 
                                             if isinstance(l, (tf.keras.layers.Dense, 
                                                             tf.keras.layers.Conv2D, 
                                                             tf.keras.layers.Conv1D))]):
                        activations[layer.name] = activations_list[i]
                        
            elif self.framework == "onnx":
                sess = ort.InferenceSession(self.model_path)
                input_name = sess.get_inputs()[0].name
                
                if isinstance(data, dict) and 'input_ids' in data:
                    input_data = data['input_ids'].numpy()
                elif isinstance(data, torch.Tensor):
                    input_data = data.numpy()
                else:
                    input_data = data
                
                outputs = sess.run(None, {input_name: input_data})
                for i, output in enumerate(outputs):
                    activations[f"output_{i}"] = output
                    
            return activations
            
        except Exception as e:
            logger.error(f"Error extracting activations: {str(e)}")
            return {}
    
    def _extract_weight_matrices(self) -> Dict:
        """Extract weight matrices from the model"""
        weight_matrices = {}
        
        try:
            if self.framework == "pytorch" and isinstance(self.model, torch.nn.Module):
                for name, param in self.model.named_parameters():
                    if "weight" in name:
                        weight_matrices[name] = param.data
                        
            elif self.framework == "tensorflow" and hasattr(self.model, 'layers'):
                for layer in self.model.layers:
                    if hasattr(layer, 'weights'):
                        for weight in layer.weights:
                            if "kernel" in weight.name or "weight" in weight.name:
                                weight_matrices[weight.name] = weight.numpy()
                                
            elif self.framework == "onnx":
                for initializer in self.model.graph.initializer:
                    if "weight" in initializer.name.lower() or "kernel" in initializer.name.lower():
                        weight = onnx.numpy_helper.to_array(initializer)
                        weight_matrices[initializer.name] = weight
                        
            return weight_matrices
            
        except Exception as e:
            logger.error(f"Error extracting weights: {str(e)}")
            return {}
    
    def _activation_clustering_detection(self, clean_data: Any = None) -> Dict:
        """
        Detect backdoors using activation clustering method
        """
        result = DetectionResult(
            method='activation_clustering',
            description='Detects backdoors by clustering intermediate layer activations'
        )
        
        try:
            if not HAS_SKLEARN:
                raise ImportError("scikit-learn required for activation clustering")
                
            activations = self._extract_activations(clean_data)
            if not activations:
                result.findings.append("Could not extract activations from the model")
                return result.__dict__
            
            # Process activations
            flat_activations = []
            for layer_name, layer_activations in activations.items():
                if isinstance(layer_activations, torch.Tensor):
                    layer_activations = layer_activations.cpu().numpy()
                
                if len(layer_activations.shape) > 2:
                    layer_activations = np.mean(layer_activations, axis=tuple(range(2, len(layer_activations.shape))))
                
                flat_activations.append(layer_activations)
            
            all_activations = np.concatenate(flat_activations, axis=1)
            
            # Dimensionality reduction
            if all_activations.shape[1] > 50:
                pca = PCA(n_components=50, random_state=42)
                all_activations = pca.fit_transform(all_activations)
            
            # Clustering
            n_samples = all_activations.shape[0]
            if n_samples < MIN_SAMPLES_FOR_CLUSTERING:
                result.findings.append(f"Not enough samples for clustering (have {n_samples}, need {MIN_SAMPLES_FOR_CLUSTERING})")
                return result.__dict__
                
            n_clusters = min(5, n_samples // 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(all_activations)
            
            # Analyze clusters
            cluster_counts = np.bincount(cluster_labels)
            cluster_sizes = cluster_counts / np.sum(cluster_counts)
            small_clusters = np.where(cluster_sizes < 0.2)[0]
            
            if len(small_clusters) > 0:
                result.score = min(0.3 + 0.1 * len(small_clusters), 0.8)
                result.findings.append(f"Found {len(small_clusters)} small activation clusters")
                
                for cluster in small_clusters[:MAX_REPORT_EVIDENCE]:
                    result.evidence.append({
                        'type': 'small_cluster',
                        'cluster_id': int(cluster),
                        'size': float(cluster_sizes[cluster]),
                        'samples': int(cluster_counts[cluster])
                    })
            
            # Outlier detection
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(all_activations)
            outlier_count = np.sum(outliers == -1)
            
            if outlier_count > 0:
                outlier_ratio = outlier_count / len(outliers)
                result.score = min(result.score + 0.2 * outlier_ratio, 1.0)
                result.findings.append(f"Found {outlier_count} outlier activations ({outlier_ratio:.2%})")
                result.evidence.append({
                    'type': 'outlier_activations',
                    'count': int(outlier_count),
                    'ratio': float(outlier_ratio)
                })
                
        except Exception as e:
            result.error = str(e)
            logger.error(f"Error in activation clustering: {str(e)}")
            
        return result.__dict__
    
    def _neuron_inspection_detection(self, clean_data: Any = None) -> Dict:
        """
        Detect backdoors by inspecting neuron behavior
        """
        result = DetectionResult(
            method='neuron_inspection',
            description='Detects backdoors by analyzing individual neuron behavior'
        )
        
        try:
            activations = self._extract_activations(clean_data)
            if not activations:
                result.findings.append("Could not extract activations from the model")
                return result.__dict__
            
            suspicious_neurons = []
            
            for layer_name, layer_activations in activations.items():
                if isinstance(layer_activations, torch.Tensor):
                    layer_activations = layer_activations.cpu().numpy()
                
                # For convolutional layers
                if len(layer_activations.shape) == 4:
                    channel_activations = np.mean(layer_activations, axis=(2, 3))
                    
                    for channel_idx in range(channel_activations.shape[1]):
                        channel_data = channel_activations[:, channel_idx]
                        stats = self._analyze_neuron_behavior(channel_data)
                        
                        if stats['is_suspicious']:
                            suspicious_neurons.append({
                                'layer': layer_name,
                                'neuron_id': channel_idx,
                                'type': 'conv_channel',
                                **stats
                            })
                
                # For dense layers
                elif len(layer_activations.shape) == 2:
                    for neuron_idx in range(layer_activations.shape[1]):
                        neuron_data = layer_activations[:, neuron_idx]
                        stats = self._analyze_neuron_behavior(neuron_data)
                        
                        if stats['is_suspicious']:
                            suspicious_neurons.append({
                                'layer': layer_name,
                                'neuron_id': neuron_idx,
                                'type': 'dense_neuron',
                                **stats
                            })
            
            if suspicious_neurons:
                result.score = min(0.1 * len(suspicious_neurons), 0.8)
                result.findings.append(f"Found {len(suspicious_neurons)} suspicious neurons")
                result.evidence = suspicious_neurons[:MAX_REPORT_EVIDENCE]
                
        except Exception as e:
            result.error = str(e)
            logger.error(f"Error in neuron inspection: {str(e)}")
            
        return result.__dict__
    
    def _analyze_neuron_behavior(self, activations: np.ndarray) -> Dict:
        """Analyze behavior of a single neuron/channel"""
        mean_activation = np.mean(activations)
        max_activation = np.max(activations)
        activation_variance = np.var(activations)
        
        # Calculate sparsity
        threshold = 0.01 * max_activation if max_activation > 0 else 0
        sparsity = np.mean(activations < threshold)
        
        # Check for suspicious patterns
        is_sparse = sparsity > 0.8 and max_activation > 10 * mean_activation
        is_high_var = activation_variance > 10 * (mean_activation ** 2 + 1e-6)
        
        return {
            'mean_activation': float(mean_activation),
            'max_activation': float(max_activation),
            'variance': float(activation_variance),
            'sparsity': float(sparsity),
            'is_suspicious': is_sparse or is_high_var,
            'flags': ['sparse'] if is_sparse else [] + ['high_variance'] if is_high_var else []
        }
    
    def _adversarial_analysis_detection(self, clean_data: Any = None) -> Dict:
        """
        Detect backdoors using adversarial analysis
        """
        result = DetectionResult(
            method='adversarial_analysis',
            description='Detects backdoors by testing model sensitivity to adversarial perturbations'
        )
        
        try:
            if not HAS_SKLEARN:
                raise ImportError("scikit-learn required for adversarial analysis")
                
            if clean_data is None:
                clean_data = self._generate_synthetic_data()
            
            if isinstance(clean_data, dict) and 'input_ids' in clean_data and HAS_PYTORCH:
                input_ids = clean_data['input_ids']
                attention_mask = clean_data.get('attention_mask', torch.ones_like(input_ids))
                
                if hasattr(self.model, 'get_input_embeddings'):
                    embeddings = self.model.get_input_embeddings()(input_ids)
                    perturbation = torch.randn_like(embeddings) * 0.01
                    perturbed_embeddings = embeddings + perturbation
                    
                    with torch.no_grad():
                        original_outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
                        perturbed_outputs = self.model(inputs_embeds=perturbed_embeddings, attention_mask=attention_mask)
                        
                        if hasattr(original_outputs, 'logits'):
                            original_logits = original_outputs.logits
                            perturbed_logits = perturbed_outputs.logits
                        else:
                            original_logits = original_outputs[0]
                            perturbed_outputs = perturbed_outputs[0]
                            
                    output_diff = torch.mean(torch.abs(original_logits - perturbed_logits)).item()
                    
                    if output_diff > 1.0:
                        result.score = min(0.2 + 0.1 * output_diff, 0.8)
                        result.findings.append(f"Model shows high sensitivity to perturbations (diff: {output_diff:.4f})")
                        result.evidence.append({
                            'type': 'high_sensitivity',
                            'output_diff': float(output_diff)
                        })
            
            elif isinstance(clean_data, (torch.Tensor, np.ndarray)):
                if isinstance(clean_data, np.ndarray):
                    data_tensor = torch.tensor(clean_data, dtype=torch.float32)
                else:
                    data_tensor = clean_data
                    
                perturbation = torch.randn_like(data_tensor) * 0.01
                perturbed_data = data_tensor + perturbation
                
                if self.framework == "pytorch":
                    with torch.no_grad():
                        original_outputs = self.model(data_tensor)
                        perturbed_outputs = self.model(perturbed_data)
                        
                        if hasattr(original_outputs, 'logits'):
                            original_logits = original_outputs.logits
                            perturbed_logits = perturbed_outputs.logits
                        else:
                            original_logits = original_outputs[0]
                            perturbed_logits = perturbed_outputs[0]
                            
                    output_diff = torch.mean(torch.abs(original_logits - perturbed_logits)).item()
                
                elif self.framework == "tensorflow":
                    original_outputs = self.model(data_tensor.numpy())
                    perturbed_outputs = self.model(perturbed_data.numpy())
                    
                    original_logits = original_outputs[0] if isinstance(original_outputs, list) else original_outputs
                    perturbed_logits = perturbed_outputs[0] if isinstance(perturbed_outputs, list) else perturbed_outputs
                    
                    original_logits = torch.tensor(original_logits)
                    perturbed_logits = torch.tensor(perturbed_logits)
                    output_diff = torch.mean(torch.abs(original_logits - perturbed_logits)).item()
                
                if output_diff > 1.0:
                    result.score = min(0.2 + 0.1 * output_diff, 0.8)
                    result.findings.append(f"Model shows high sensitivity to perturbations (diff: {output_diff:.4f})")
                    result.evidence.append({
                        'type': 'high_sensitivity',
                        'output_diff': float(output_diff)
                    })
                    
        except Exception as e:
            result.error = str(e)
            logger.error(f"Error in adversarial analysis: {str(e)}")
            
        return result.__dict__
    
    def _spectral_analysis_detection(self) -> Dict:
        """
        Detect backdoors using spectral analysis of weights
        """
        result = DetectionResult(
            method='spectral_analysis',
            description='Detects backdoors by analyzing spectral properties of model weights'
        )
        
        try:
            if not HAS_SKLEARN:
                raise ImportError("scikit-learn required for spectral analysis")
                
            weight_matrices = self._extract_weight_matrices()
            if not weight_matrices:
                result.findings.append("Could not extract weight matrices from the model")
                return result.__dict__
            
            spectral_anomalies = []
            
            for layer_name, weights in weight_matrices.items():
                if isinstance(weights, torch.Tensor):
                    weights = weights.cpu().numpy()
                
                # For 2D weight matrices
                if len(weights.shape) == 2:
                    s = np.linalg.svd(weights, compute_uv=False)
                    s_norm = s / np.sum(s)
                    spectral_entropy = entropy(s_norm)
                    spectral_norm = np.max(s)
                    effective_rank = np.exp(spectral_entropy)
                    
                    if spectral_entropy < 0.5:
                        spectral_anomalies.append({
                            'layer': layer_name,
                            'type': 'low_entropy',
                            'entropy': float(spectral_entropy),
                            'effective_rank': float(effective_rank)
                        })
                    
                    if spectral_norm > 100:
                        spectral_anomalies.append({
                            'layer': layer_name,
                            'type': 'high_norm',
                            'norm': float(spectral_norm)
                        })
                
                # For 4D weight tensors (Conv layers)
                elif len(weights.shape) == 4:
                    out_channels, in_channels, k_h, k_w = weights.shape
                    weights_2d = weights.reshape(out_channels, -1)
                    s = np.linalg.svd(weights_2d, compute_uv=False)
                    s_norm = s / np.sum(s)
                    spectral_entropy = entropy(s_norm)
                    spectral_norm = np.max(s)
                    
                    if spectral_entropy < 0.5:
                        spectral_anomalies.append({
                            'layer': layer_name,
                            'type': 'low_entropy',
                            'entropy': float(spectral_entropy)
                        })
                    
                    if spectral_norm > 100:
                        spectral_anomalies.append({
                            'layer': layer_name,
                            'type': 'high_norm',
                            'norm': float(spectral_norm)
                        })
            
            if spectral_anomalies:
                result.score = min(0.15 * len(spectral_anomalies), 0.7)
                result.findings.append(f"Found {len(spectral_anomalies)} spectral anomalies")
                result.evidence = spectral_anomalies[:MAX_REPORT_EVIDENCE]
                
        except Exception as e:
            result.error = str(e)
            logger.error(f"Error in spectral analysis: {str(e)}")
            
        return result.__dict__
    
    def _input_output_analysis_detection(self, trigger_patterns: List[str] = None) -> Dict:
        """
        Detect backdoors by analyzing input-output patterns
        """
        result = DetectionResult(
            method='input_output_analysis',
            description='Detects backdoors by testing model behavior with potential trigger patterns'
        )
        
        try:
            trigger_patterns = trigger_patterns or self._generate_trigger_patterns()
            suspicious_patterns = []
            
            for pattern in trigger_patterns:
                if self.model_type == ModelType.TEXT and self.tokenizer:
                    if isinstance(pattern, str):
                        inputs = self.tokenizer(pattern, return_tensors="pt", padding=True, truncation=True)
                    else:
                        inputs = pattern
                    
                    with torch.no_grad():
                        if self.framework == "pytorch":
                            outputs = self.model(**inputs)
                            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                            predictions = torch.argmax(logits, dim=-1)
                            confidence = torch.softmax(logits, dim=-1)[0].max().item()
                            prediction = predictions[0].item()
                        elif self.framework == "tensorflow":
                            outputs = self.model({
                                'input_ids': inputs['input_ids'].numpy(),
                                'attention_mask': inputs.get('attention_mask', torch.ones_like(inputs['input_ids'])).numpy()
                            })
                            logits = outputs[0] if isinstance(outputs, list) else outputs
                            predictions = tf.argmax(logits, axis=-1).numpy()
                            confidence = tf.nn.softmax(logits)[0].numpy().max()
                            prediction = predictions[0]
                    
                    if confidence > 0.95:
                        suspicious_patterns.append({
                            'pattern': pattern if isinstance(pattern, str) else "text_pattern",
                            'prediction': int(prediction),
                            'confidence': float(confidence),
                            'type': 'high_confidence'
                        })
                
                elif self.model_type == ModelType.VISION:
                    if isinstance(pattern, np.ndarray):
                        pattern_tensor = torch.tensor(pattern, dtype=torch.float32)
                    else:
                        pattern_tensor = pattern
                    
                    with torch.no_grad():
                        if self.framework == "pytorch":
                            outputs = self.model(pattern_tensor)
                            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                            predictions = torch.argmax(logits, dim=-1)
                            confidence = torch.softmax(logits, dim=-1)[0].max().item()
                            prediction = predictions[0].item()
                        elif self.framework == "tensorflow":
                            outputs = self.model(pattern_tensor.numpy())
                            logits = outputs[0] if isinstance(outputs, list) else outputs
                            predictions = tf.argmax(logits, axis=-1).numpy()
                            confidence = tf.nn.softmax(logits)[0].numpy().max()
                            prediction = predictions[0]
                    
                    if confidence > 0.95:
                        suspicious_patterns.append({
                            'pattern': "image_pattern",
                            'prediction': int(prediction),
                            'confidence': float(confidence),
                            'type': 'high_confidence'
                        })
            
            if suspicious_patterns:
                result.score = min(0.2 * len(suspicious_patterns), 0.8)
                result.findings.append(f"Found {len(suspicious_patterns)} suspicious trigger patterns")
                result.evidence = suspicious_patterns[:MAX_REPORT_EVIDENCE]
                
        except Exception as e:
            result.error = str(e)
            logger.error(f"Error in input-output analysis: {str(e)}")
            
        return result.__dict__
    
    def _generate_trigger_patterns(self) -> List:
        """Generate potential trigger patterns for testing"""
        patterns = []
        
        if self.model_type == ModelType.TEXT:
            text_triggers = [
                "The backdoor trigger is activated",
                "Special command: execute backdoor",
                "Hidden pattern: backdoor",
                "Trigger word: backdoor",
                "Backdoor activation sequence",
                "Execute malicious payload",
                "Secret command: attack",
                "Activate hidden behavior",
                "Malicious input pattern",
                "Exploit trigger sequence"
            ]
            patterns.extend(text_triggers)
            
            # Add some random text patterns
            patterns.extend([
                "".join(np.random.choice(list("abcdefghijklmnopqrstuvwxyz"), size=20)) for _ in range(3)
            ])
        
        elif self.model_type == ModelType.VISION:
            # Generate image patterns with potential triggers
            for _ in range(5):  # Small patches in corners
                img = np.random.rand(3, 224, 224).astype(np.float32)
                corner = np.random.choice(4)
                if corner == 0:
                    img[:, :10, :10] = 1.0
                elif corner == 1:
                    img[:, :10, -10:] = 1.0
                elif corner == 2:
                    img[:, -10:, :10] = 1.0
                else:
                    img[:, -10:, -10:] = 1.0
                patterns.append(img)
            
            for _ in range(3):  # Specific color channels
                img = np.random.rand(3, 224, 224).astype(np.float32)
                channel = np.random.randint(0, 3)
                img[channel, :, :] = 1.0
                patterns.append(img)
            
            # Add some random noise patterns
            patterns.extend([
                np.random.rand(3, 224, 224).astype(np.float32) for _ in range(2)
            ])
        
        return patterns
    
    def _weight_analysis_detection(self) -> Dict:
        """
        Detect backdoors by analyzing model weights
        """
        result = DetectionResult(
            method='weight_analysis',
            description='Detects backdoors by analyzing model weight distributions'
        )
        
        try:
            if not HAS_SKLEARN:
                raise ImportError("scikit-learn required for weight analysis")
                
            weight_matrices = self._extract_weight_matrices()
            if not weight_matrices:
                result.findings.append("Could not extract weight matrices from the model")
                return result.__dict__
            
            weight_anomalies = []
            
            for layer_name, weights in weight_matrices.items():
                if isinstance(weights, torch.Tensor):
                    weights = weights.cpu().numpy()
                
                flat_weights = weights.flatten()
                mean_weight = np.mean(flat_weights)
                std_weight = np.std(flat_weights)
                max_weight = np.max(np.abs(flat_weights))
                
                # Check for anomalies
                if std_weight < 1e-6:
                    weight_anomalies.append({
                        'layer': layer_name,
                        'type': 'low_variance',
                        'std': float(std_weight),
                        'mean': float(mean_weight)
                    })
                
                if max_weight > 100:
                    weight_anomalies.append({
                        'layer': layer_name,
                        'type': 'large_weights',
                        'max_weight': float(max_weight)
                    })
                
                # Check sparsity
                sparsity = np.mean(np.abs(flat_weights) < 1e-6)
                if sparsity > 0.9:
                    weight_anomalies.append({
                        'layer': layer_name,
                        'type': 'high_sparsity',
                        'sparsity': float(sparsity)
                    })
                
                # Check for bimodal distribution
                if len(flat_weights) > 100:
                    hist, _ = np.histogram(flat_weights, bins=20)
                    peak_count = np.sum((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))
                    if peak_count >= 2:
                        weight_anomalies.append({
                            'layer': layer_name,
                            'type': 'bimodal_distribution',
                            'peak_count': int(peak_count)
                        })
            
            if weight_anomalies:
                result.score = min(0.15 * len(weight_anomalies), 0.7)
                result.findings.append(f"Found {len(weight_anomalies)} weight anomalies")
                result.evidence = weight_anomalies[:MAX_REPORT_EVIDENCE]
                
        except Exception as e:
            result.error = str(e)
            logger.error(f"Error in weight analysis: {str(e)}")
            
        return result.__dict__
    
    def detect_backdoors(self, methods: List[str] = None, data_path: str = None, 
                        trigger_patterns: List[str] = None) -> Dict:
        """
        Run backdoor detection using specified methods
        
        Args:
            methods: List of detection methods to use
            data_path: Path to clean dataset for comparison
            trigger_patterns: List of potential trigger patterns to test
            
        Returns:
            Dictionary containing detection results
        """
        if methods is None:
            methods = DEFAULT_DETECTION_METHODS
        
        self.detection_results['methods_used'] = methods
        logger.info(f"Starting backdoor detection with methods: {', '.join(methods)}")
        
        # Prepare data for analysis
        clean_data = self._prepare_data(data_path)
        
        # Run detection methods
        for method in methods:
            method_start = time.time()
            result = None
            
            try:
                if method == 'activation_clustering':
                    result = self._activation_clustering_detection(clean_data)
                elif method == 'neuron_inspection':
                    result = self._neuron_inspection_detection(clean_data)
                elif method == 'adversarial_analysis':
                    result = self._adversarial_analysis_detection(clean_data)
                elif method == 'spectral_analysis':
                    result = self._spectral_analysis_detection()
                elif method == 'input_output_analysis':
                    result = self._input_output_analysis_detection(trigger_patterns)
                elif method == 'weight_analysis':
                    result = self._weight_analysis_detection()
                else:
                    logger.warning(f"Method {method} not available or missing dependencies")
                    continue
                
                # Calculate execution time
                exec_time = time.time() - method_start
                result['execution_time'] = exec_time
                
                self.detection_results['findings'].append(result)
                self.detection_results['backdoor_score'] += result.get('score', 0)
                
                logger.info(f"Completed {method} with score: {result.get('score', 0):.2f} (took {exec_time:.2f}s)")
                
            except Exception as e:
                logger.error(f"Error in {method} detection: {str(e)}")
                self.detection_results['findings'].append({
                    'method': method,
                    'error': str(e),
                    'score': 0.0,
                    'execution_time': time.time() - method_start
                })
        
        # Normalize backdoor score
        if methods:
            valid_methods = [m for m in methods if m in DEFAULT_DETECTION_METHODS]
            if valid_methods:
                self.detection_results['backdoor_score'] /= len(valid_methods)
        
        # Determine confidence level
        score = self.detection_results['backdoor_score']
        if score > 0.7:
            self.detection_results['confidence'] = ConfidenceLevel.HIGH.name.lower()
        elif score > 0.4:
            self.detection_results['confidence'] = ConfidenceLevel.MEDIUM.name.lower()
        else:
            self.detection_results['confidence'] = ConfidenceLevel.LOW.name.lower()
        
        logger.info(f"Detection completed. Overall score: {score:.2f}, Confidence: {self.detection_results['confidence']}")
        return self.detection_results
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on detection results"""
        recommendations = []
        score = self.detection_results.get('backdoor_score', 0)
        
        # General recommendations based on score
        if score > 0.7:
            recommendations.extend([
                "HIGH RISK: Backdoor likely present. Do not use this model in production.",
                "Consider retraining the model from scratch with verified clean data.",
                "If model must be used, implement strict input validation and sanitization.",
                "Monitor model behavior closely in production for unexpected outputs.",
                "Consider reporting this finding to relevant security teams."
            ])
        elif score > 0.4:
            recommendations.extend([
                "MODERATE RISK: Potential backdoor detected. Exercise caution.",
                "Perform additional validation with trusted datasets.",
                "Implement input filtering to detect potential trigger patterns.",
                "Consider fine-tuning the model with clean data to mitigate risks.",
                "Monitor model behavior in staging before production deployment."
            ])
        else:
            recommendations.extend([
                "LOW RISK: No significant backdoor detected.",
                "Implement standard model monitoring and auditing procedures.",
                "Keep the model and its dependencies updated.",
                "Follow secure deployment practices for AI models.",
                "Periodically re-scan the model for backdoors."
            ])
        
        # Method-specific recommendations
        method_findings = defaultdict(list)
        for finding in self.detection_results.get('findings', []):
            method = finding.get('method', '')
            method_findings[method].append(finding)
        
        if method_findings.get('activation_clustering'):
            recommendations.append(
                "Activation clustering found suspicious patterns. Consider analyzing model "
                "activations on clean vs. potentially poisoned data."
            )
        
        if method_findings.get('neuron_inspection'):
            recommendations.append(
                "Suspicious neurons detected. Inspect these neurons for potential backdoor behavior "
                "and consider neuron pruning or regularization."
            )
        
        if method_findings.get('adversarial_analysis'):
            recommendations.append(
                "Model shows sensitivity to perturbations. Consider adversarial training "
                "to improve model robustness."
            )
        
        if method_findings.get('spectral_analysis'):
            recommendations.append(
                "Spectral anomalies detected in weights. Review weight matrices with "
                "unusual spectral properties."
            )
        
        if method_findings.get('input_output_analysis'):
            recommendations.append(
                "Suspicious trigger patterns found. Expand testing with more trigger patterns "
                "and implement input filtering."
            )
        
        if method_findings.get('weight_analysis'):
            recommendations.append(
                "Weight anomalies detected. Consider weight pruning or regularization "
                "to address potential backdoors."
            )
        
        return recommendations
    
    def generate_report(self, output_path: str = None, format: str = 'json') -> str:
        """Generate comprehensive backdoor detection report"""
        if not output_path:
            output_path = f"{Path(self.model_path).stem}_backdoor_report"
        
        # Add summary to the results
        self.detection_results['summary'] = {
            'overall_backdoor_score': self.detection_results.get('backdoor_score', 0),
            'confidence': self.detection_results.get('confidence', 'low'),
            'recommendations': self._generate_recommendations()
        }
        
        # Generate report in specified format
        if format.lower() == 'json':
            output_path = f"{output_path}.json"
            with open(output_path, 'w') as f:
                json.dump(self.detection_results, f, indent=2)
            logger.info(f"JSON report saved to {output_path}")
        elif format.lower() == 'html':
            output_path = f"{output_path}.html"
            self._generate_html_report(output_path)
            logger.info(f"HTML report saved to {output_path}")
        else:
            raise ValueError(f"Unsupported report format: {format}")
        
        return output_path
    
    def _generate_html_report(self, output_path: str):
        """Generate an HTML version of the report"""
        try:
            from jinja2 import Environment, FileSystemLoader
            import markdown
            
            # Create a simple HTML template if not available
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Backdoor Detection Report</title>
                <style>
                    body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }
                    .container { max-width: 1200px; margin: 0 auto; }
                    h1, h2, h3 { color: #2c3e50; }
                    .card { background: #f9f9f9; border-radius: 5px; padding: 15px; margin-bottom: 20px; }
                    .score-high { color: #e74c3c; font-weight: bold; }
                    .score-medium { color: #f39c12; font-weight: bold; }
                    .score-low { color: #27ae60; font-weight: bold; }
                    table { width: 100%; border-collapse: collapse; margin: 15px 0; }
                    th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                    tr:hover { background-color: #f5f5f5; }
                    .evidence { font-family: monospace; font-size: 0.9em; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Backdoor Detection Report</h1>
                    <p>Generated at: {{ report.timestamp }}</p>
                    
                    <div class="card">
                        <h2>Model Information</h2>
                        <p><strong>Path:</strong> {{ report.model_path }}</p>
                        <p><strong>Source:</strong> {{ report.model_source }}</p>
                        <p><strong>Type:</strong> {{ report.model_type }}</p>
                        <p><strong>Framework:</strong> {{ report.framework }}</p>
                    </div>
                    
                    <div class="card">
                        <h2>Summary</h2>
                        <p>
                            <strong>Overall Backdoor Score:</strong> 
                            <span class="score-{{ report.confidence }}">{{ "%.2f"|format(report.summary.overall_backdoor_score) }}</span>
                            (Confidence: {{ report.confidence }})
                        </p>
                        
                        <h3>Recommendations</h3>
                        <ul>
                            {% for rec in report.summary.recommendations %}
                                <li>{{ rec }}</li>
                            {% endfor %}
                        </ul>
                    </div>
                    
                    <div class="card">
                        <h2>Detailed Findings</h2>
                        {% for finding in report.findings %}
                            <div style="margin-bottom: 20px;">
                                <h3>{{ finding.method.replace('_', ' ')|title }} (Score: {{ "%.2f"|format(finding.score) }})</h3>
                                <p>{{ finding.description }}</p>
                                
                                {% if finding.error %}
                                    <p style="color: red;">Error: {{ finding.error }}</p>
                                {% endif %}
                                
                                {% if finding.findings %}
                                    <h4>Findings:</h4>
                                    <ul>
                                        {% for f in finding.findings %}
                                            <li>{{ f }}</li>
                                        {% endfor %}
                                    </ul>
                                {% endif %}
                                
                                {% if finding.evidence %}
                                    <h4>Evidence (Top {{ MAX_EVIDENCE }}):</h4>
                                    <table>
                                        <thead>
                                            <tr>
                                                {% for key in finding.evidence[0].keys() %}
                                                    <th>{{ key|title }}</th>
                                                {% endfor %}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {% for item in finding.evidence %}
                                                <tr>
                                                    {% for value in item.values() %}
                                                        <td>{{ value }}</td>
                                                    {% endfor %}
                                                </tr>
                                            {% endfor %}
                                        </tbody>
                                    </table>
                                {% endif %}
                            </div>
                        {% endfor %}
                    </div>
                    
                    <div class="card">
                        <h2>System Information</h2>
                        <p><strong>Platform:</strong> {{ report.system_info.platform }}</p>
                        <p><strong>Python Version:</strong> {{ report.system_info.python_version }}</p>
                        <p><strong>CPU:</strong> {{ report.system_info.cpu }}</p>
                        <p><strong>Report Time:</strong> {{ report.system_info.system_time }}</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Create a temporary directory for the template
            with tempfile.TemporaryDirectory() as temp_dir:
                template_path = Path(temp_dir) / "report_template.html"
                with open(template_path, 'w') as f:
                    f.write(html_template)
                
                env = Environment(loader=FileSystemLoader(temp_dir))
                template = env.get_template("report_template.html")
                
                # Render the template with our data
                html_output = template.render(
                    report=self.detection_results,
                    MAX_EVIDENCE=MAX_REPORT_EVIDENCE
                )
                
                # Write to file
                with open(output_path, 'w') as f:
                    f.write(html_output)
                    
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {str(e)}")
            raise
    
    def visualize_results(self, output_path: str = None, format: str = 'png') -> str:
        """Generate visualization of detection results"""
        if not HAS_VISUALIZATION:
            raise ImportError("Visualization libraries not available. Install matplotlib and seaborn.")
        
        if not output_path:
            output_path = f"{Path(self.model_path).stem}_backdoor_visualization"
        
        try:
            # Determine file extension
            ext = format.lower()
            if ext not in ['png', 'pdf', 'svg']:
                ext = 'png'
                logger.warning(f"Unsupported format {format}, defaulting to PNG")
            
            output_path = f"{output_path}.{ext}"
            
            # Create figure with subplots
            plt.figure(figsize=(15, 12))
            plt.suptitle(f'Backdoor Detection Results for {Path(self.model_path).name}', fontsize=16)
            
            # Create grid for subplots
            grid = plt.GridSpec(3, 2, hspace=0.4, wspace=0.3)
            
            # Plot 1: Backdoor score by method
            ax1 = plt.subplot(grid[0, 0])
            methods = []
            scores = []
            for finding in self.detection_results.get('findings', []):
                if 'method' in finding and 'score' in finding:
                    methods.append(finding['method'])
                    scores.append(finding['score'])
            
            if methods:
                colors = []
                for score in scores:
                    if score > 0.7:
                        colors.append('#e74c3c')  # red
                    elif score > 0.4:
                        colors.append('#f39c12')  # orange
                    else:
                        colors.append('#27ae60')  # green
                
                bars = ax1.bar(methods, scores, color=colors)
                ax1.set_title('Backdoor Score by Detection Method')
                ax1.set_ylabel('Score')
                ax1.set_ylim(0, 1)
                plt.sca(ax1)
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}',
                            ha='center', va='bottom')
            
            # Plot 2: Evidence types
            ax2 = plt.subplot(grid[0, 1])
            evidence_types = defaultdict(int)
            for finding in self.detection_results.get('findings', []):
                for evidence in finding.get('evidence', []):
                    evidence_type = evidence.get('type', 'unknown')
                    evidence_types[evidence_type] += 1
            
            if evidence_types:
                labels = list(evidence_types.keys())
                sizes = list(evidence_types.values())
                explode = [0.1] * len(labels)  # explode all slices slightly
                
                _, texts, autotexts = ax2.pie(sizes, explode=explode, labels=labels, 
                                             autopct='%1.1f%%', startangle=90)
                ax2.set_title('Distribution of Evidence Types')
                
                # Make labels more readable
                for text in texts + autotexts:
                    text.set_fontsize(8)
            
            # Plot 3: Overall backdoor score gauge
            ax3 = plt.subplot(grid[1, 0])
            overall_score = self.detection_results.get('backdoor_score', 0)
            confidence = self.detection_results.get('confidence', 'low')
            
            # Create gauge
            theta = np.linspace(0, np.pi, 100)
            r = np.ones(100)
            
            # Color segments
            ax3.fill_between(theta, 0, r, where=(theta < np.pi/3), color='#27ae60', alpha=0.3)  # green
            ax3.fill_between(theta, 0, r, where=(theta >= np.pi/3) & (theta < 2*np.pi/3), color='#f39c12', alpha=0.3)  # orange
            ax3.fill_between(theta, 0, r, where=(theta >= 2*np.pi/3), color='#e74c3c', alpha=0.3)  # red
            
            # Needle position
            needle_pos = np.pi - overall_score * np.pi
            ax3.plot([needle_pos, needle_pos], [0, 1], color='black', linewidth=2)
            
            # Add labels
            ax3.text(np.pi/6, 1.1, 'Low', ha='center')
            ax3.text(np.pi/2, 1.1, 'Medium', ha='center')
            ax3.text(5*np.pi/6, 1.1, 'High', ha='center')
            
            # Add score text
            ax3.text(0, 0, f"Score: {overall_score:.2f}\n({confidence})", 
                     ha='center', va='center', fontsize=12)
            
            ax3.set_title('Overall Backdoor Score')
            ax3.axis('off')
            
            # Plot 4: Recommendations word cloud
            ax4 = plt.subplot(grid[1, 1])
            recommendations = self._generate_recommendations()
            
            if HAS_WORDCLOUD and recommendations:
                text = ' '.join(recommendations)
                wordcloud = WordCloud(width=400, height=200, background_color='white').generate(text)
                ax4.imshow(wordcloud, interpolation='bilinear')
                ax4.axis('off')
                ax4.set_title('Key Recommendations')
            elif recommendations:
                ax4.text(0.1, 0.9, "Top Recommendations:", fontsize=10, fontweight='bold')
                for i, rec in enumerate(recommendations[:5]):
                    ax4.text(0.1, 0.8 - i*0.15, f"• {rec}", fontsize=8, wrap=True)
                ax4.axis('off')
                ax4.set_title('Recommendations (WordCloud not available)')
            
            # Plot 5: Method execution times
            ax5 = plt.subplot(grid[2, :])
            methods = []
            times = []
            for finding in self.detection_results.get('findings', []):
                if 'method' in finding and 'execution_time' in finding:
                    methods.append(finding['method'])
                    times.append(finding['execution_time'])
            
            if methods:
                ax5.barh(methods, times, color='#3498db')
                ax5.set_title('Method Execution Times (seconds)')
                ax5.set_xlabel('Time (s)')
                ax5.grid(axis='x', linestyle='--', alpha=0.6)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualization saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to generate visualization: {str(e)}")
            raise

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="BackdoorScanner - Advanced AI Model Backdoor Detection Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("model_path", 
                       help="Path to the model or model identifier (for Hugging Face/ModelScope)")
    
    parser.add_argument("-s", "--source", 
                       choices=["huggingface", "modelscope", "local"], 
                       default="local",
                       help="Model source")
    
    parser.add_argument("-t", "--type", 
                       choices=["text", "vision", "multimodal", "auto"], 
                       default="auto",
                       help="Model type")
    
    parser.add_argument("-f", "--framework", 
                       choices=["pytorch", "tensorflow", "onnx", "auto"], 
                       default="auto",
                       help="Model framework")
    
    parser.add_argument("-m", "--methods", 
                       nargs='+', 
                       choices=DEFAULT_DETECTION_METHODS,
                       help="Detection methods to use")
    
    parser.add_argument("-d", "--data", 
                       help="Path to clean dataset for comparison")
    
    parser.add_argument("-p", "--patterns", 
                       nargs='+', 
                       help="Custom trigger patterns to test")
    
    parser.add_argument("-o", "--output", 
                       help="Output directory for reports and visualizations")
    
    parser.add_argument("-r", "--report-format", 
                       choices=["json", "html"], 
                       default="json",
                       help="Report output format")
    
    parser.add_argument("-v", "--visualization-format", 
                       choices=["png", "pdf", "svg"], 
                       default="png",
                       help="Visualization output format")
    
    parser.add_argument("-V", "--verbose", 
                       action="store_true", 
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    try:
        # Initialize BackdoorDetector
        detector = BackdoorDetector(
            model_path=args.model_path,
            model_source=args.source,
            model_type=args.type,
            framework=args.framework,
            verbose=args.verbose
        )
        
        # Run backdoor detection
        results = detector.detect_backdoors(
            methods=args.methods,
            data_path=args.data,
            trigger_patterns=args.patterns
        )
        
        # Create output directory if specified
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            report_base = Path(args.model_path).stem
            report_path = os.path.join(args.output, f"{report_base}_backdoor_report.{args.report_format}")
            viz_path = os.path.join(args.output, f"{report_base}_backdoor_visualization.{args.visualization_format}")
        else:
            report_path = None
            viz_path = None
        
        # Generate report
        detector.generate_report(report_path, format=args.report_format)
        
        # Generate visualization
        try:
            detector.visualize_results(viz_path, format=args.visualization_format)
        except Exception as e:
            if args.verbose:
                logger.error(f"Could not generate visualization: {str(e)}")
        
        # Print summary
        print("\n" + "="*60)
        print("BACKDOOR DETECTION SUMMARY".center(60))
        print("="*60)
        print(f"Model: {args.model_path}")
        print(f"Source: {args.source}")
        print(f"Type: {detector.model_type}")
        print(f"Framework: {detector.framework}")
        print(f"Overall Backdoor Score: {results.get('backdoor_score', 0):.4f}")
        print(f"Confidence: {results.get('confidence', 'low').upper()}")
        print(f"Methods Used: {', '.join(results.get('methods_used', []))}")
        
        if results.get('findings'):
            print("\nFINDINGS:")
            for finding in results.get('findings', []):
                method = finding.get('method', 'unknown')
                score = finding.get('score', 0)
                findings_list = finding.get('findings', [])
                print(f"- {method.replace('_', ' ').title()}: {score:.4f}")
                for f in findings_list[:3]:  # Show top 3 findings per method
                    print(f"  * {f}")
        
        if results.get('summary', {}).get('recommendations'):
            print("\nRECOMMENDATIONS:")
            for rec in results.get('summary', {}).get('recommendations', [])[:5]:
                print(f"- {rec}")
        
        print("\n" + "="*60)
        
        # Exit with appropriate code
        if results.get('backdoor_score', 0) > 0.7:
            print("HIGH RISK: Backdoor detected! Do not use this model in production.")
            sys.exit(1)
        elif results.get('backdoor_score', 0) > 0.4:
            print("MODERATE RISK: Potential backdoor detected. Exercise caution.")
            sys.exit(2)
        else:
            print("LOW RISK: No significant backdoor detected.")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(3)

if __name__ == "__main__":
    main()
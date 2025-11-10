"""
Example script for scGPT data integration fine-tuning.

This example demonstrates how to use scGPT for batch correction and data integration
tasks, implementing the key objectives from the scGPT integration tutorial:
- GEPC (Gene Expression Modeling for Cell)
- ECS (Elastic Cell Similarity) 
- DAR (Domain Adversarial Regularization)
- DSBN (Domain-Specific Batch Normalization)
"""

import numpy as np
import scanpy as sc
from helical.models.scgpt import scGPTFineTuningModel, scGPTConfig
from helical.models.fine_tune.data_integration_head import DataIntegrationHead
import anndata as ad
from sklearn.metrics import adjusted_rand_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_integration_data(adata: ad.AnnData, batch_key: str = "batch") -> tuple:
    """
    Prepare data for integration training.
    
    Parameters
    ----------
    adata : ad.AnnData
        The annotated data matrix with batch information
    batch_key : str
        Column name in adata.obs containing batch labels
        
    Returns
    -------
    tuple
        (processed_adata, batch_labels, num_batches)
    """
    logger.info("Preparing data for integration training...")
    
    # Ensure batch labels are categorical and create integer encoding
    adata.obs[f"str_{batch_key}"] = adata.obs[batch_key].astype(str)
    batch_id_labels = adata.obs[f"str_{batch_key}"].astype("category").cat.codes.values
    adata.obs["batch_id"] = batch_id_labels
    
    num_batches = len(adata.obs[f"str_{batch_key}"].unique())
    logger.info(f"Found {num_batches} batches in dataset")
    
    # Basic preprocessing for scGPT
    # Note: In practice, you should also filter genes by vocabulary
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Calculate QC metrics
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    
    # Normalization
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Select highly variable genes
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable].copy()
    
    logger.info(f"Preprocessed data: {adata.n_obs} cells x {adata.n_vars} genes")
    
    return adata, batch_id_labels, num_batches


def run_integration_training(
    adata: ad.AnnData,
    batch_labels: np.ndarray,
    num_batches: int,
    batch_size: int = 32,
    epochs: int = 15,
    ecs_threshold: float = 0.8,
    dab_weight: float = 1.0,
    mask_ratio: float = 0.4
) -> scGPTFineTuningModel:
    """
    Run data integration training with scGPT.
    
    Parameters
    ----------
    adata : ad.AnnData
        Preprocessed data
    batch_labels : np.ndarray
        Integer-encoded batch labels
    num_batches : int
        Number of unique batches
    batch_size : int
        Training batch size
    epochs : int
        Number of training epochs
    ecs_threshold : float
        Threshold for Elastic Cell Similarity
    dab_weight : float
        Weight for Domain Adversarial loss
    mask_ratio : float
        Masking ratio for MLM objective
        
    Returns
    -------
    scGPTFineTuningModel
        Trained integration model
    """
    logger.info("Setting up scGPT for data integration...")
    
    # Configure scGPT for integration
    scgpt_config = scGPTConfig(batch_size=batch_size)
    
    # Create custom data integration head
    integration_head = DataIntegrationHead(
        num_batches=num_batches,
        ecs_threshold=ecs_threshold,
        dab_weight=dab_weight,
        use_dsbn=True,
        dropout=0.2
    )
    
    # Create fine-tuning model
    scgpt_model = scGPTFineTuningModel(
        scGPT_config=scgpt_config,
        fine_tuning_head=integration_head,
        output_size=None  # Not needed when passing head instance
    )
    
    # Process data for scGPT
    logger.info("Processing data for scGPT...")
    dataset = scgpt_model.process_data(adata)
    
    # Train the integration model
    logger.info("Starting integration training...")
    scgpt_model.train_data_integration(
        train_input_data=dataset,
        train_batch_labels=batch_labels,
        epochs=epochs,
        mask_ratio=mask_ratio,
        ecs_weight=10.0,
        dab_weight=dab_weight,
        optimizer_params={"lr": 1e-4},
        lr_scheduler_params={
            'name': 'linear',
            'num_warmup_steps': 0,
            'num_training_steps': len(dataset) // batch_size * epochs
        }
    )
    
    logger.info("Integration training completed!")
    return scgpt_model


def evaluate_integration(
    model: scGPTFineTuningModel,
    adata: ad.AnnData,
    batch_labels: np.ndarray,
    cell_type_key: str = "cell_type"
) -> dict:
    """
    Evaluate integration quality.
    
    Parameters
    ----------
    model : scGPTFineTuningModel
        Trained integration model
    adata : ad.AnnData
        Original data with cell type annotations
    batch_labels : np.ndarray
        Batch labels
    cell_type_key : str
        Column name for cell type annotations
        
    Returns
    -------
    dict
        Integration quality metrics
    """
    logger.info("Evaluating integration quality...")
    
    # Get integrated embeddings
    dataset = model.process_data(adata)
    outputs = model.get_outputs(dataset)
    
    # If using DataIntegrationHead, extract embeddings from the output dict
    if isinstance(outputs, dict) and 'embeddings' in outputs:
        embeddings = outputs['embeddings']
    else:
        embeddings = outputs
    
    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Store embeddings in adata
    adata.obsm["X_scGPT_integrated"] = embeddings
    
    # Compute neighbors and UMAP for visualization
    sc.pp.neighbors(adata, use_rep="X_scGPT_integrated")
    sc.tl.umap(adata)
    
    # Basic integration metrics
    metrics = {}
    
    # Batch mixing: measure how well batches are mixed
    if "batch_id" in adata.obs.columns:
        # Simplified batch mixing score (lower is better integrated)
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=10).fit(embeddings)
        distances, indices = knn.kneighbors(embeddings)
        
        batch_mixing_score = 0
        for i in range(len(batch_labels)):
            neighbors_batches = batch_labels[indices[i]]
            same_batch_neighbors = np.sum(neighbors_batches == batch_labels[i])
            batch_mixing_score += same_batch_neighbors / 10
        
        metrics["batch_mixing_score"] = batch_mixing_score / len(batch_labels)
        logger.info(f"Batch mixing score: {metrics['batch_mixing_score']:.3f} (lower is better)")
    
    # Cell type preservation: measure if cell types are preserved
    if cell_type_key in adata.obs.columns:
        # Silhouette score for cell type clustering
        from sklearn.metrics import silhouette_score
        cell_type_labels = adata.obs[cell_type_key].astype('category').cat.codes
        sil_score = silhouette_score(embeddings, cell_type_labels)
        metrics["cell_type_preservation"] = sil_score
        logger.info(f"Cell type preservation (silhouette): {sil_score:.3f} (higher is better)")
        
        # ARI score if both batch and cell type are available
        if "batch_id" in adata.obs.columns:
            # This is a simplified metric - in practice you'd use more sophisticated methods
            ari_batch = adjusted_rand_score(batch_labels, cell_type_labels)
            metrics["batch_celltype_ari"] = ari_batch
            logger.info(f"Batch-celltype ARI: {ari_batch:.3f}")
    
    return metrics


def main():
    """
    Main example function demonstrating data integration workflow.
    """
    logger.info("Starting scGPT Data Integration Example")
    
    # Example with simulated data - replace with your dataset
    logger.info("Loading example dataset...")
    
    # For this example, let's create some dummy data
    # In practice, you would load your real multi-batch dataset
    n_obs, n_vars = 1000, 2000
    X = np.random.negative_binomial(10, 0.3, size=(n_obs, n_vars))
    
    # Create mock batch labels (3 batches)
    batch_labels = np.random.choice([0, 1, 2], size=n_obs)
    cell_types = np.random.choice(['TypeA', 'TypeB', 'TypeC', 'TypeD'], size=n_obs)
    
    # Create AnnData object
    adata = ad.AnnData(X=X.astype(float))
    adata.obs['batch'] = [f'batch_{i}' for i in batch_labels]
    adata.obs['cell_type'] = cell_types
    adata.var_names = [f'gene_{i}' for i in range(n_vars)]
    adata.var['gene_name'] = adata.var_names  # scGPT expects this column
    
    logger.info(f"Created example dataset: {adata.n_obs} cells x {adata.n_vars} genes")
    
    # Prepare data for integration
    adata_processed, batch_id_labels, num_batches = prepare_integration_data(adata, "batch")
    
    # Run integration training
    model = run_integration_training(
        adata=adata_processed,
        batch_labels=batch_id_labels,
        num_batches=num_batches,
        batch_size=16,  # Small batch size for example
        epochs=5,       # Few epochs for example
        ecs_threshold=0.8,
        dab_weight=1.0,
        mask_ratio=0.4
    )
    
    # Evaluate integration
    metrics = evaluate_integration(
        model=model,
        adata=adata_processed,
        batch_labels=batch_id_labels,
        cell_type_key="cell_type"
    )
    
    logger.info("Integration complete! Summary metrics:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")
    
    # Save the trained model
    logger.info("Saving trained model...")
    model.save_model("scgpt_integration_model.pth")
    
    logger.info("Example completed successfully!")
    
    return model, adata_processed, metrics


if __name__ == "__main__":
    model, adata, metrics = main()
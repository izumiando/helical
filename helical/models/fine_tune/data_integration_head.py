import torch
import torch.nn.functional as F
from typing import Dict, Optional
from helical.models.fine_tune.fine_tuning_heads import HelicalBaseFineTuningHead


class DataIntegrationHead(HelicalBaseFineTuningHead):
    """Data Integration Head for batch correction fine-tuning.
    
    This head implements the objectives from the scGPT integration tutorial:
    - GEPC (Gene Expression Modeling for Cell): Standard masked language modeling
    - ECS (Elastic Cell Similarity): Threshold-based cell similarity objective
    - DAR (Domain Adversarial Regularization): Batch correction via adversarial training
    - DSBN (Domain-Specific Batch Normalization): Separate batch norms for different batches
    
    Parameters
    ----------
    num_batches : int
        Number of different batches in the dataset
    embedding_dim : int
        Dimension of input embeddings (set automatically via set_dim_size)
    ecs_threshold : float, default=0.8
        Threshold for elastic cell similarity (0.0-1.0, 0.0 to disable)
    dab_weight : float, default=1.0
        Weight for domain adversarial batch correction loss
    use_dsbn : bool, default=True
        Whether to use domain-specific batch normalization
    dropout : float, default=0.2
        Dropout rate
    """
    
    def __init__(
        self, 
        num_batches: int,
        ecs_threshold: float = 0.8,
        dab_weight: float = 1.0, 
        use_dsbn: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_batches = num_batches
        self.ecs_threshold = ecs_threshold
        self.dab_weight = dab_weight
        self.use_dsbn = use_dsbn
        self.dropout_rate = dropout
        
        # Will be set via set_dim_size
        self.embedding_dim = None
        
    def forward(self, 
                embeddings: torch.Tensor, 
                batch_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for data integration.
        
        Parameters
        ----------
        embeddings : torch.Tensor
            Cell embeddings from the foundation model [batch_size, embedding_dim]
        batch_labels : Optional[torch.Tensor]
            Batch labels for each cell [batch_size]
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing integration outputs and intermediate results
        """
        batch_size = embeddings.size(0)
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        # Domain-specific batch normalization if enabled
        if self.use_dsbn and batch_labels is not None:
            embeddings = self.dsbn(embeddings, batch_labels)
        elif hasattr(self, 'batch_norm'):
            embeddings = self.batch_norm(embeddings)
            
        outputs = {'embeddings': embeddings}
        
        # Elastic Cell Similarity (ECS) if enabled
        if self.ecs_threshold > 0:
            ecs_loss = self._compute_ecs_loss(embeddings)
            outputs['ecs_loss'] = ecs_loss
            
        # Domain Adversarial Regularization (DAR) for batch correction
        if batch_labels is not None:
            # Reverse gradients for adversarial training
            reversed_embeddings = self._reverse_gradient(embeddings)
            batch_predictions = self.batch_discriminator(reversed_embeddings)
            outputs['batch_predictions'] = batch_predictions
            
        return outputs
    
    def _compute_ecs_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute Elastic Cell Similarity loss."""
        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise similarities
        similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
        
        # Apply threshold - cells above threshold should be similar
        threshold_mask = similarity_matrix > self.ecs_threshold
        
        # ECS loss: encourage high similarity for pairs above threshold
        ecs_loss = F.relu(self.ecs_threshold - similarity_matrix) * threshold_mask.float()
        
        return ecs_loss.mean()
    
    def _reverse_gradient(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Reverse gradients for adversarial training (gradient reversal layer)."""
        return GradientReversalFunction.apply(x, alpha)
        
    def set_dim_size(self, dim_size: int) -> None:
        """Set up layers based on input embedding dimensions."""
        self.embedding_dim = dim_size
        
        # Dropout layer
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)
        
        # Domain-specific batch normalization or regular batch norm
        if self.use_dsbn:
            self.dsbn = DomainSpecificBatchNorm1d(dim_size, self.num_batches)
        else:
            self.batch_norm = torch.nn.BatchNorm1d(dim_size)
            
        # Batch discriminator for adversarial training
        self.batch_discriminator = torch.nn.Sequential(
            torch.nn.Linear(dim_size, dim_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout_rate),
            torch.nn.Linear(dim_size // 2, self.num_batches)
        )


class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial training."""
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DomainSpecificBatchNorm1d(torch.nn.Module):
    """Domain-Specific Batch Normalization for different batches."""
    
    def __init__(self, num_features: int, num_domains: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.num_domains = num_domains
        self.eps = eps
        self.affine = affine
        
        # Create separate batch norms for each domain/batch
        self.batch_norms = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(num_features, eps=eps, affine=affine)
            for _ in range(num_domains)
        ])
        
    def forward(self, x: torch.Tensor, domain_labels: torch.Tensor) -> torch.Tensor:
        """
        Apply domain-specific batch normalization.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor [batch_size, num_features]
        domain_labels : torch.Tensor
            Domain/batch labels for each sample [batch_size]
            
        Returns
        -------
        torch.Tensor
            Normalized tensor
        """
        if self.training:
            # During training, apply appropriate batch norm for each sample
            outputs = []
            for domain_id in range(self.num_domains):
                mask = domain_labels == domain_id
                if mask.sum() > 0:
                    domain_input = x[mask]
                    # Handle single sample case by temporarily setting batch norm to eval mode
                    if mask.sum() == 1:
                        was_training = self.batch_norms[domain_id].training
                        self.batch_norms[domain_id].eval()
                        domain_output = self.batch_norms[domain_id](domain_input)
                        if was_training:
                            self.batch_norms[domain_id].train()
                    else:
                        domain_output = self.batch_norms[domain_id](domain_input)
                    outputs.append((mask, domain_output))
            
            # Reconstruct the full batch
            result = torch.zeros_like(x)
            for mask, output in outputs:
                result[mask] = output
            return result
        else:
            # During evaluation, use the first batch norm (or could be made configurable)
            return self.batch_norms[0](x)
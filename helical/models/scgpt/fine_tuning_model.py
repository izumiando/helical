from typing import Literal, Optional, Dict
from helical.models.scgpt.data_collator import DataCollator
from helical.models.scgpt.dataset import Dataset
import torch
from torch import optim
from torch.nn.modules import loss
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import get_scheduler
from helical.models.base_models import HelicalBaseFineTuningHead
from helical.models.scgpt import scGPT, scGPTConfig
from helical.models.base_models import HelicalBaseFineTuningModel
from helical.models.fine_tune.data_integration_head import DataIntegrationHead
import logging
import numpy as np

logger = logging.getLogger(__name__)


class scGPTFineTuningModel(HelicalBaseFineTuningModel, scGPT):
    """Fine-tuning model for the scGPT model.

    Example
    ----------
    ```python
    from helical.models.scgpt import scGPTFineTuningModel, scGPTConfig

    # Load the desired dataset
    adata = ad.read_h5ad("dataset.h5ad")

    # Get the desired label class
    cell_types = list(ann_data.obs.cell_type)

    # Get unique labels
    label_set = set(cell_types)

    # Create the fine-tuning model with the relevant configs
    scgpt_config=scGPTConfig(batch_size=10)
    scgpt_fine_tune = scGPTFineTuningModel(scGPT_config=scgpt_config, fine_tuning_head="classification", output_size=len(label_set))

    # Process the data for training
    data = scgpt_fine_tune.process_data(adata)

    # Create a dictionary mapping the classes to unique integers for training
    class_id_dict = dict(zip(label_set, [i for i in range(len(label_set))]))

    for i in range(len(cell_types)):
        cell_types[i] = class_id_dict[cell_types[i]]

    # Fine-tune
    scgpt_fine_tune.train(train_input_data=dataset, train_labels=cell_types)
    ```

    Parameters
    ----------
    scGPT_config : scGPTConfig
        The scGPT configs for fine-tuning model, the same configs that would be used to instantiate the standard scGPT model.
    fine_tuning_head : Literal["classification", "regression", "data_integration"] | HelicalBaseFineTuningHead
        The fine-tuning head that is appended to the model. This can either be a string (options available: "classification", "regression", "data_integration") specifying the task or a custom fine-tuning head inheriting from HelicalBaseFineTuningHead.
    output_size : Optional[int]
        The output size of the fine-tuning model. This is required if the fine_tuning_head is a string specified task. For a classification task this is number of unique classes.

    Methods
    -------
    train(train_input_data: Dataset, train_labels: np.ndarray, validation_input_data: Optional[Dataset], validation_labels: Optional[np.ndarray], optimizer: optim, optimizer_params: dict, loss_function: loss, epochs: int, lr_scheduler_params: Optional[dict])
        Fine-tunes the scGPT model with different head modules.
    get_outputs(dataset: Dataset) -> np.ndarray
        Get the outputs of the fine-tuned model.

    """

    def __init__(
        self,
        scGPT_config: scGPTConfig,
        fine_tuning_head: Literal["classification", "regression", "data_integration"] | HelicalBaseFineTuningHead,
        output_size: Optional[int] = None,
    ):
        HelicalBaseFineTuningModel.__init__(self, fine_tuning_head, output_size)
        scGPT.__init__(self, scGPT_config)

        self.fine_tuning_head.set_dim_size(self.config["embsize"])

    def _forward(
        self,
        input_gene_ids: torch.Tensor,
        data_dict: dict,
        src_key_padding_mask: torch.Tensor,
        use_batch_labels: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Forward method of the fine-tuning model.

        Parameters
        ----------
        input_gene_ids : torch.Tensor
            The input tensor to the fine-tuning model.
        data_dict : dict
            The data dictionary containing the expression data and batch labels.
        src_key_padding_mask : torch.Tensor
            The source key padding mask tensor.
        use_batch_labels : bool
            Whether to use batch labels.
        device : torch.device
            The device to run the model on.

        Returns
        -------
        torch.Tensor
            The output tensor of the fine-tuning model.
        """
        embeddings = self.model._encode(
            input_gene_ids,
            data_dict["expr"].to(device),
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=(
                data_dict["batch_labels"].to(device) if use_batch_labels else None
            ),
        )

        if self.config["emb_mode"] == "cls":
            embeddings = embeddings[:, 0, :]
        else:
            embeddings = embeddings[:, 1:, :].mean(dim=1)

        # Handle different head types
        if isinstance(self.fine_tuning_head, DataIntegrationHead):
            # For data integration, pass batch labels to the head
            batch_labels = data_dict.get("batch_labels") if use_batch_labels else None
            output = self.fine_tuning_head(embeddings, batch_labels)
        else:
            # Standard forward for classification/regression heads
            output = self.fine_tuning_head(embeddings)
        return output

    def train(
        self,
        train_input_data: Dataset,
        train_labels: np.ndarray,
        validation_input_data=None,
        validation_labels=None,
        optimizer: optim = optim.AdamW,
        optimizer_params: dict = {"lr": 0.0001},
        loss_function: loss = loss.CrossEntropyLoss(),
        epochs: int = 1,
        # freeze_layers: int = 0,
        lr_scheduler_params: Optional[dict] = None,
    ):
        """Fine-tunes the scGPT model with different head modules.

        Parameters
        ----------
        train_input_data : Dataset
            A helical scGPT processed dataset for fine-tuning
        train_labels : ndarray
            The labels for the training data. These should be stored as unique per class integers.
        validation_input_data : Dataset, default = None
            A helical scGPT processed dataset for per epoch validation. If this is not specified, no validation will be performed.
        validation_labels : ndarray, default = None
            The labels for the validation data. These should be stored as unique per class integers.
        optimizer : torch.optim, default = torch.optim.AdamW
            The optimizer to be used for training.
        optimizer_params : dict
            The optimizer parameters to be used for the optimizer specified. This list should NOT include model parameters.
            e.g. optimizer_params = {'lr': 0.0001}
        loss_function : torch.nn.modules.loss, default = torch.nn.modules.loss.CrossEntropyLoss()
            The loss function to be used.
        epochs : int, optional, default = 10
            The number of epochs to train the model
        lr_scheduler_params : dict, default = None
            The learning rate scheduler parameters for the transformers get_scheduler method. The optimizer will be taken from the optimizer input and should not be included in the learning scheduler parameters. If not specified, no scheduler will be used.
            e.g. lr_scheduler_params = { 'name': 'linear', 'num_warmup_steps': 0, 'num_training_steps': 5 }
        """

        device = next(self.model.parameters()).device

        try:
            use_batch_labels = train_input_data.batch_ids is not None
        except:
            use_batch_labels = False

        collator = DataCollator(
            do_padding=True,
            pad_token_id=self.vocab[self.config["pad_token"]],
            pad_value=self.config["pad_value"],
            do_mlm=False,
            do_binning=True,
            max_length=1200,
            sampling=True,
            keep_first_n_tokens=1,
        )

        data_loader = DataLoader(
            train_input_data,
            batch_size=self.config["batch_size"],
            sampler=SequentialSampler(train_input_data),
            collate_fn=collator,
            drop_last=False,
            pin_memory=True,
        )

        if validation_input_data is not None:
            validation_data_loader = DataLoader(
                validation_input_data,
                batch_size=self.config["batch_size"],
                sampler=SequentialSampler(validation_input_data),
                collate_fn=collator,
                drop_last=False,
                pin_memory=True,
            )

        self.to(device)
        self.model.train()
        self.fine_tuning_head.train()
        optimizer = optimizer(self.parameters(), **optimizer_params)

        lr_scheduler = None
        if lr_scheduler_params is not None:
            lr_scheduler = get_scheduler(optimizer=optimizer, **lr_scheduler_params)

        logger.info("Starting Fine-Tuning")
        for j in range(epochs):
            batch_count = 0
            batch_loss = 0.0
            batches_processed = 0
            training_loop = tqdm(data_loader)
            for data_dict in training_loop:
                input_gene_ids = data_dict["gene"].to(device)
                src_key_padding_mask = input_gene_ids.eq(
                    self.vocab[self.config["pad_token"]]
                )
                output = self._forward(
                    input_gene_ids,
                    data_dict,
                    src_key_padding_mask,
                    use_batch_labels,
                    device,
                )
                labels = torch.tensor(
                    train_labels[batch_count : batch_count + self.config["batch_size"]],
                    device=device,
                )
                batch_count += self.config["batch_size"]
                loss = loss_function(output, labels)
                loss.backward()
                batch_loss += loss.item()
                batches_processed += 1
                optimizer.step()
                optimizer.zero_grad()

                training_loop.set_postfix({"loss": batch_loss / batches_processed})
                training_loop.set_description(f"Fine-Tuning: epoch {j+1}/{epochs}")

            if lr_scheduler is not None:
                lr_scheduler.step()

            if validation_input_data is not None:
                testing_loop = tqdm(
                    validation_data_loader, desc="Fine-Tuning Validation"
                )
                val_loss = 0.0
                count = 0.0
                validation_batch_count = 0
                for validation_data_dict in testing_loop:
                    input_gene_ids = validation_data_dict["gene"].to(device)
                    src_key_padding_mask = input_gene_ids.eq(
                        self.vocab[self.config["pad_token"]]
                    )
                    output = self._forward(
                        input_gene_ids,
                        validation_data_dict,
                        src_key_padding_mask,
                        use_batch_labels,
                        device,
                    )
                    val_labels = torch.tensor(
                        validation_labels[
                            validation_batch_count : validation_batch_count
                            + self.config["batch_size"]
                        ],
                        device=device,
                    )
                    val_loss += loss_function(output, val_labels).item()
                    validation_batch_count += self.config["batch_size"]
                    count += 1.0
                    testing_loop.set_postfix({"val_loss": val_loss / count})
        logger.info(f"Fine-Tuning Complete. Epochs: {epochs}")

    def train_data_integration(
        self,
        train_input_data: Dataset,
        train_batch_labels: np.ndarray,
        validation_input_data=None,
        validation_batch_labels=None,
        optimizer: optim = optim.AdamW,
        optimizer_params: dict = {"lr": 0.0001},
        epochs: int = 15,
        mask_ratio: float = 0.4,
        ecs_weight: float = 10.0,
        dab_weight: float = 1.0,
        lr_scheduler_params: Optional[dict] = None,
    ):
        """Fine-tunes scGPT for data integration (batch correction).
        
        This method implements the integration training from the scGPT tutorial with
        multiple objectives: GEPC, ECS, and DAR (Domain Adversarial Regularization).

        Parameters
        ----------
        train_input_data : Dataset
            A helical scGPT processed dataset for fine-tuning
        train_batch_labels : ndarray
            The batch labels for the training data (integer encoded).
        validation_input_data : Dataset, default=None
            A helical scGPT processed dataset for per epoch validation.
        validation_batch_labels : ndarray, default=None
            The batch labels for the validation data.
        optimizer : torch.optim, default=torch.optim.AdamW
            The optimizer to be used for training.
        optimizer_params : dict
            The optimizer parameters.
        epochs : int, default=15
            The number of epochs to train the model
        mask_ratio : float, default=0.4
            Proportion of genes to mask for MLM objective
        ecs_weight : float, default=10.0
            Weight for Elastic Cell Similarity loss
        dab_weight : float, default=1.0
            Weight for Domain Adversarial Batch correction loss
        lr_scheduler_params : dict, default=None
            The learning rate scheduler parameters.
        """
        if not isinstance(self.fine_tuning_head, DataIntegrationHead):
            raise ValueError("train_data_integration requires DataIntegrationHead")
            
        device = next(self.model.parameters()).device
        
        # Enable batch labels usage for data integration
        self.model.use_batch_labels = True
        
        # Configure data collator for integration training
        collator = DataCollator(
            do_padding=True,
            pad_token_id=self.vocab[self.config["pad_token"]],
            pad_value=self.config["pad_value"],
            do_mlm=True,  # Enable masked language modeling
            mlm_probability=mask_ratio,
            do_binning=True,
            max_length=1200,
            sampling=True,
            keep_first_n_tokens=1,
        )

        data_loader = DataLoader(
            train_input_data,
            batch_size=self.config["batch_size"],
            sampler=SequentialSampler(train_input_data),
            collate_fn=collator,
            drop_last=False,
            pin_memory=True,
        )

        if validation_input_data is not None:
            validation_data_loader = DataLoader(
                validation_input_data,
                batch_size=self.config["batch_size"],
                sampler=SequentialSampler(validation_input_data),
                collate_fn=collator,
                drop_last=False,
                pin_memory=True,
            )

        self.to(device)
        self.model.train()
        self.fine_tuning_head.train()
        optimizer = optimizer(self.parameters(), **optimizer_params)

        # Loss functions
        mse_criterion = torch.nn.MSELoss()
        ce_criterion = torch.nn.CrossEntropyLoss()

        lr_scheduler = None
        if lr_scheduler_params is not None:
            lr_scheduler = get_scheduler(optimizer=optimizer, **lr_scheduler_params)

        logger.info("Starting Data Integration Fine-Tuning")
        for epoch in range(epochs):
            batch_count = 0
            total_loss = 0.0
            mlm_loss_total = 0.0
            ecs_loss_total = 0.0  
            dab_loss_total = 0.0
            batches_processed = 0
            
            training_loop = tqdm(data_loader)
            for data_dict in training_loop:
                input_gene_ids = data_dict["gene"].to(device)
                input_values = data_dict["expr"].to(device)
                
                # Get batch labels for current batch
                current_batch_labels = torch.tensor(
                    train_batch_labels[batch_count : batch_count + input_gene_ids.size(0)],
                    device=device,
                    dtype=torch.long
                )
                
                # Create source key padding mask
                src_key_padding_mask = input_gene_ids.eq(
                    self.vocab[self.config["pad_token"]]
                )
                
                # Get embeddings from scGPT
                embeddings = self.model._encode(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=current_batch_labels,
                )
                
                # Process embeddings based on embedding mode
                if self.config["emb_mode"] == "cls":
                    embeddings = embeddings[:, 0, :]
                else:
                    embeddings = embeddings[:, 1:, :].mean(dim=1)
                
                # Forward through integration head
                outputs = self.fine_tuning_head(embeddings, current_batch_labels)
                
                # Combined loss computation
                loss = 0.0
                
                # 1. Standard MLM loss (reconstruction)
                if "mlm_output" in data_dict and data_dict["mlm_output"] is not None:
                    mlm_targets = data_dict["mlm_output"].to(device)
                    # Use embeddings to reconstruct masked tokens
                    reconstruction_loss = mse_criterion(embeddings, mlm_targets)
                    loss += reconstruction_loss
                    mlm_loss_total += reconstruction_loss.item()
                
                # 2. Elastic Cell Similarity (ECS) loss
                if "ecs_loss" in outputs:
                    ecs_loss = outputs["ecs_loss"]
                    loss += ecs_weight * ecs_loss
                    ecs_loss_total += ecs_loss.item()
                
                # 3. Domain Adversarial Regularization (DAR) loss
                if "batch_predictions" in outputs:
                    batch_pred = outputs["batch_predictions"]
                    dab_loss = ce_criterion(batch_pred, current_batch_labels)
                    loss += dab_weight * dab_loss
                    dab_loss_total += dab_loss.item()
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                batch_count += input_gene_ids.size(0)
                batches_processed += 1
                
                # Update progress bar
                training_loop.set_postfix({
                    "total_loss": total_loss / batches_processed,
                    "mlm_loss": mlm_loss_total / batches_processed,
                    "ecs_loss": ecs_loss_total / batches_processed if ecs_loss_total > 0 else 0,
                    "dab_loss": dab_loss_total / batches_processed if dab_loss_total > 0 else 0,
                })
                training_loop.set_description(f"Integration Training: epoch {epoch+1}/{epochs}")

            if lr_scheduler is not None:
                lr_scheduler.step()

            # Validation
            if validation_input_data is not None:
                self.model.eval()
                self.fine_tuning_head.eval()
                
                val_loss = 0.0
                val_count = 0
                validation_batch_count = 0
                
                testing_loop = tqdm(validation_data_loader, desc="Validation")
                with torch.no_grad():
                    for validation_data_dict in testing_loop:
                        input_gene_ids = validation_data_dict["gene"].to(device)
                        input_values = validation_data_dict["expr"].to(device)
                        
                        current_val_batch_labels = torch.tensor(
                            validation_batch_labels[validation_batch_count : validation_batch_count + input_gene_ids.size(0)],
                            device=device,
                            dtype=torch.long
                        )
                        
                        src_key_padding_mask = input_gene_ids.eq(
                            self.vocab[self.config["pad_token"]]
                        )
                        
                        embeddings = self.model._encode(
                            input_gene_ids,
                            input_values,
                            src_key_padding_mask=src_key_padding_mask,
                            batch_labels=current_val_batch_labels,
                        )
                        
                        if self.config["emb_mode"] == "cls":
                            embeddings = embeddings[:, 0, :]
                        else:
                            embeddings = embeddings[:, 1:, :].mean(dim=1)
                        
                        outputs = self.fine_tuning_head(embeddings, current_val_batch_labels)
                        
                        # Compute validation loss (simplified)
                        batch_val_loss = 0.0
                        if "ecs_loss" in outputs:
                            batch_val_loss += ecs_weight * outputs["ecs_loss"]
                        if "batch_predictions" in outputs:
                            batch_pred = outputs["batch_predictions"]
                            batch_val_loss += dab_weight * ce_criterion(batch_pred, current_val_batch_labels)
                        
                        val_loss += batch_val_loss.item()
                        validation_batch_count += input_gene_ids.size(0)
                        val_count += 1.0
                        testing_loop.set_postfix({"val_loss": val_loss / val_count})
                
                self.model.train()
                self.fine_tuning_head.train()
                
        logger.info(f"Data Integration Fine-Tuning Complete. Epochs: {epochs}")

    def get_outputs(
        self,
        dataset: Dataset,
    ) -> np.ndarray:
        """Get the outputs of the fine-tuned model.

        Parameters
        ----------
        dataset : Dataset
            The dataset to get the outputs from.

        Returns
        -------
        np.ndarray
            The outputs of the fine-tuned model.
        """
        device = next(self.model.parameters()).device
        self.to(device)
        self.model.eval()
        self.fine_tuning_head.eval()
        
        # fix seeds
        np.random.seed(self.config["binning_seed"])
        torch.manual_seed(self.config["binning_seed"])

        try:
            use_batch_labels = dataset.batch_ids is not None
        except:
            use_batch_labels = False
        
        # Temporarily set model's batch label usage for inference
        original_use_batch_labels = self.model.use_batch_labels
        self.model.use_batch_labels = use_batch_labels

        collator = DataCollator(
            do_padding=True,
            pad_token_id=self.vocab[self.config["pad_token"]],
            pad_value=self.config["pad_value"],
            do_mlm=False,
            do_binning=True,
            max_length=1200,
            sampling=True,
            keep_first_n_tokens=1,
        )

        data_loader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            sampler=SequentialSampler(dataset),
            collate_fn=collator,
            drop_last=False,
            pin_memory=True,
        )

        testing_loop = tqdm(data_loader, desc="Fine-Tuning Validation")
        outputs = []
        for validation_data_dict in testing_loop:
            input_gene_ids = validation_data_dict["gene"].to(device)
            src_key_padding_mask = input_gene_ids.eq(
                self.vocab[self.config["pad_token"]]
            )
            output = self._forward(
                input_gene_ids,
                validation_data_dict,
                src_key_padding_mask,
                use_batch_labels,
                device,
            )
            # Handle different output types from different heads
            if isinstance(output, dict):
                # DataIntegrationHead returns a dict with embeddings
                if 'embeddings' in output:
                    embeddings = output['embeddings'].detach().cpu().numpy()
                else:
                    raise ValueError("DataIntegrationHead output missing 'embeddings' key")
            else:
                # Standard heads return tensors directly
                embeddings = output.detach().cpu().numpy()
            outputs.append(embeddings)

        # Restore original batch label setting
        self.model.use_batch_labels = original_use_batch_labels
        
        return np.vstack(outputs)

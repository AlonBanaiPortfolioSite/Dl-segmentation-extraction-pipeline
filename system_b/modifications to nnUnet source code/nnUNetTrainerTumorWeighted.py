from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch


class nnUNetTrainerTumorWeighted(nnUNetTrainer):
    """
    nnU-Net trainer with 3× weight for tumor class (class 1)
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        """
        Initialize with correct signature
        """
        super().__init__(plans, configuration, fold, dataset_json, device)
        print("\n" + "=" * 70)
        print("🔥 USING TUMOR-WEIGHTED LOSS 🔥")
        print("Class weights: [background=1.0, tumor=3.0, organoid=1.0]")
        print("=" * 70 + "\n")
    
    def _build_loss(self):
        """
        Modify the parent's loss to add class weights
        """
        # Get the parent's loss (which is already wrapped with DeepSupervisionWrapper)
        loss = super()._build_loss()
        
        # The parent returns DeepSupervisionWrapper wrapping DC_and_CE_loss
        # We need to modify the inner DC_and_CE_loss's CE component
        
        # Access the wrapped loss
        inner_loss = loss.loss  # This is the DC_and_CE_loss
        
        # Create class weights
        class_weights = torch.tensor([1.0, 3.0, 1.0])
        if self.device.type == 'cuda':
            class_weights = class_weights.cuda()
        
        # Modify the CE loss to use weights
        inner_loss.ce.weight = class_weights
        
        return loss
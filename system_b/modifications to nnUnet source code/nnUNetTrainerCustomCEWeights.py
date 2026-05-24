from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch

class nnUNetTrainerCustomCEWeights(nnUNetTrainer):
    """
    Base trainer for custom CrossEntropy class weights.
    Subclasses override class_weights to test different weight configurations.
    """
    
    # Subclasses override this: [background_weight, tumor_weight, organoid_weight]
    class_weights = [1.0, 1.0, 1.0]  # Default: no weighting
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        """
        Initialize with correct signature
        """
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        print("\n" + "=" * 70)
        print(f"🔥 USING CUSTOM CE WEIGHTS: {self.__class__.__name__} 🔥")
        print(f"Class weights: [bg={self.class_weights[0]}, tumor={self.class_weights[1]}, organoid={self.class_weights[2]}]")
        print("=" * 70 + "\n")
    
    def _build_loss(self):
        """
        Modify the parent's loss to add class weights
        """
        # Get the parent's loss (wrapped with DeepSupervisionWrapper)
        loss = super()._build_loss()
        
        # Access the inner DC_and_CE_loss
        inner_loss = loss.loss
        
        # Create class weights tensor
        weights = torch.tensor(self.class_weights, dtype=torch.float32)
        if self.device.type == 'cuda':
            weights = weights.cuda()
        
        # Modify the CE loss to use weights
        inner_loss.ce.weight = weights
        
        return loss


# ============================================================================
# TUMOR WEIGHT EXPERIMENTS (keeping class 0 and 2 at 1.0)
# ============================================================================
class nnUNetTrainerTumor1_1(nnUNetTrainerCustomCEWeights):
    class_weights = [1.0, 1.1, 1.0]
class nnUNetTrainerTumor1_2(nnUNetTrainerCustomCEWeights):
    class_weights = [1.0, 1.2, 1.0]
class nnUNetTrainerTumor1_25(nnUNetTrainerCustomCEWeights):
    class_weights = [1.0, 1.25, 1.0]
class nnUNetTrainerTumor1_3(nnUNetTrainerCustomCEWeights):
    class_weights = [1.0, 1.3, 1.0]
class nnUNetTrainerTumor1_4(nnUNetTrainerCustomCEWeights):
    class_weights = [1.0, 1.4, 1.0]
class nnUNetTrainerTumor1_5(nnUNetTrainerCustomCEWeights):
    """Tumor weight = 1.5× (conservative test)"""
    class_weights = [1.0, 1.5, 1.0]

class nnUNetTrainerTumor2(nnUNetTrainerCustomCEWeights):
    """Tumor weight = 2× (moderate)"""
    class_weights = [1.0, 2.0, 1.0]

class nnUNetTrainerTumor3(nnUNetTrainerCustomCEWeights):
    """Tumor weight = 3× (original experiment)"""
    class_weights = [1.0, 3.0, 1.0]

class nnUNetTrainerTumor5(nnUNetTrainerCustomCEWeights):
    """Tumor weight = 5× (aggressive)"""
    class_weights = [1.0, 5.0, 1.0]


# ============================================================================
# ORGANOID WEIGHT EXPERIMENTS (keeping class 0 and 1 at 1.0)
# ============================================================================

class nnUNetTrainerOrganoid1_5(nnUNetTrainerCustomCEWeights):
    """Organoid weight = 1.5× (conservative test)"""
    class_weights = [1.0, 1.0, 1.5]

class nnUNetTrainerOrganoid2(nnUNetTrainerCustomCEWeights):
    """Organoid weight = 2× (moderate)"""
    class_weights = [1.0, 1.0, 2.0]

class nnUNetTrainerOrganoid3(nnUNetTrainerCustomCEWeights):
    """Organoid weight = 3× (original experiment)"""
    class_weights = [1.0, 1.0, 3.0]

class nnUNetTrainerOrganoid5(nnUNetTrainerCustomCEWeights):
    """Organoid weight = 5× (aggressive)"""
    class_weights = [1.0, 1.0, 5.0]


# ============================================================================
# COMBINED EXPERIMENTS (adjust both tumor and organoid)
# ============================================================================

class nnUNetTrainerTumor2Organoid1_5(nnUNetTrainerCustomCEWeights):
    """Tumor=2×, Organoid=1.5× (both increased)"""
    class_weights = [1.0, 2.0, 1.5]

class nnUNetTrainerTumor1_5Organoid2(nnUNetTrainerCustomCEWeights):
    """Tumor=1.5×, Organoid=2× (inverse ratio)"""
    class_weights = [1.0, 1.5, 2.0]


# ============================================================================
# USAGE EXAMPLES
# ============================================================================
"""
Each trainer automatically creates its own results folder:

# Test if small tumor weight has opposite effect
nnUNetv2_train 305 2d 0 -tr nnUNetTrainerTumor1_5
→ Results in: nnUNetTrainerTumor1_5__nnUNetPlans__2d/fold_0/

# Test moderate tumor weight
nnUNetv2_train 305 2d 0 -tr nnUNetTrainerTumor2
→ Results in: nnUNetTrainerTumor2__nnUNetPlans__2d/fold_0/

# Compare to original 3× experiment
nnUNetv2_train 305 2d 0 -tr nnUNetTrainerTumor3
→ Results in: nnUNetTrainerTumor3__nnUNetPlans__2d/fold_0/

All experiments are kept separate - no overwriting!

To add a new weight configuration, just add:

class nnUNetTrainerTumor2_5(nnUNetTrainerCustomCEWeights):
    class_weights = [1.0, 2.5, 1.0]
"""
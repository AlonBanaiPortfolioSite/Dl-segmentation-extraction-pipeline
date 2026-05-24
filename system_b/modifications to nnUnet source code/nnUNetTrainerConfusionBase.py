from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_CE_and_Confusion_loss
import torch


class nnUNetTrainerConfusionBase(nnUNetTrainer):
    """
    Base trainer for confusion-aware loss with asymmetric penalties.
    
    This trainer uses DC_CE_and_Confusion_loss which combines:
    - Dice loss (standard segmentation overlap)
    - Cross-Entropy loss (pixel-wise classification)
    - Confusion-aware loss (asymmetric penalties for tumor FN vs organoid FP)
    
    Subclasses should override the confusion_weight class variable to test different weights.
    
    The confusion loss specifically penalizes:
    - Tumor FN (tumor → organoid): 5× weight (dangerous clinical error)
    - Organoid FP (organoid → tumor): 1× weight (acceptable false alarm)
    """
    
    # Subclasses override this to test different weights
    confusion_weight = 0.1  # Default: conservative start
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        """
        Initialize with correct signature
        """
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        print("\n" + "=" * 80)
        print(f"🔥 USING CONFUSION-AWARE LOSS 🔥")
        print(f"Trainer: {self.__class__.__name__}")
        print(f"Confusion weight: {self.confusion_weight}")
        print(f"Tumor FN penalty: 5× (missing tumor is dangerous)")
        print(f"Organoid FP penalty: 1× (false alarm is acceptable)")
        print("=" * 80 + "\n")
    
    def _build_loss(self):
        """
        Build the combined Dice + CE + Confusion loss
        Following the same pattern as OrganoidWeighted - modify parent's loss
        """
        # Get the parent's loss (which is already wrapped with DeepSupervisionWrapper)
        loss = super()._build_loss()
        
        # The parent returns DeepSupervisionWrapper wrapping DC_and_CE_loss
        # We need to REPLACE the inner loss with our DC_CE_and_Confusion_loss
        
        # Access the wrapped loss
        inner_loss = loss.loss  # This is the DC_and_CE_loss
        
        # Get the parameters from the existing loss
        soft_dice_kwargs = {
            'batch_dice': inner_loss.dc.batch_dice,
            'smooth': inner_loss.dc.smooth,
            'do_bg': inner_loss.dc.do_bg,
            'ddp': inner_loss.dc.ddp
        }
        
        ce_kwargs = {}
        if inner_loss.ce.weight is not None:
            ce_kwargs['weight'] = inner_loss.ce.weight
        
        # Create our new combined loss with same parameters
        new_loss = DC_CE_and_Confusion_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            confusion_kwargs={
                'tumor_fn_weight': 5.0,
                'organoid_fp_weight': 1.0,
                'tumor_class': 1,
                'organoid_class': 2,
                'smooth': 1e-8
            },
            weight_ce=inner_loss.weight_ce,
            weight_dice=inner_loss.weight_dice,
            weight_confusion=self.confusion_weight,  # Use the class variable!
            ignore_label=inner_loss.ignore_label
        )
        
        # Replace the inner loss in the DeepSupervisionWrapper
        loss.loss = new_loss
        
        return loss


# ============================================================================
# WEIGHT-SPECIFIC SUBCLASSES
# Each subclass only overrides the confusion_weight - all logic is inherited
# ============================================================================
class nnUNetTrainerConfusion013(nnUNetTrainerConfusionBase):
    confusion_weight = 0.13
class nnUNetTrainerConfusion012(nnUNetTrainerConfusionBase):
    confusion_weight = 0.12
class nnUNetTrainerConfusion011(nnUNetTrainerConfusionBase):
    confusion_weight = 0.11
class nnUNetTrainerConfusion0075(nnUNetTrainerConfusionBase):
    confusion_weight = 0.075
class nnUNetTrainerConfusion005(nnUNetTrainerConfusionBase):
    confusion_weight = 0.05
class nnUNetTrainerConfusion01(nnUNetTrainerConfusionBase):
    """
    Confusion-aware trainer with weight=0.1 (very conservative)
    
    Expected behavior:
    - Confusion adds ~13% penalty on top of Dice+CE during mid-training
    - Gentle nudge toward reducing tumor FN
    - Minimal risk of predicting everything as tumor
    """
    confusion_weight = 0.1


class nnUNetTrainerConfusion015(nnUNetTrainerConfusionBase):
    """
    Confusion-aware trainer with weight=0.15 (conservative)
    
    Expected behavior:
    - Confusion adds ~19% penalty on top of Dice+CE during mid-training
    - Moderate push toward reducing tumor FN
    - Low risk of predicting everything as tumor
    """
    confusion_weight = 0.15


class nnUNetTrainerConfusion02(nnUNetTrainerConfusionBase):
    """
    Confusion-aware trainer with weight=0.2 (balanced)
    
    Expected behavior:
    - Confusion adds ~26% penalty on top of Dice+CE during mid-training
    - Strong push toward reducing tumor FN
    - Some risk of increased tumor FP (false alarms)
    """
    confusion_weight = 0.2


class nnUNetTrainerConfusion025(nnUNetTrainerConfusionBase):
    """
    Confusion-aware trainer with weight=0.25 (aggressive)
    
    Expected behavior:
    - Confusion adds ~32% penalty on top of Dice+CE during mid-training
    - Very strong push toward reducing tumor FN
    - Higher risk of increased tumor FP (false alarms)
    """
    confusion_weight = 0.25


class nnUNetTrainerConfusion03(nnUNetTrainerConfusionBase):
    """
    Confusion-aware trainer with weight=0.3 (very aggressive)
    
    Expected behavior:
    - Confusion adds ~39% penalty on top of Dice+CE during mid-training
    - Maximum push toward reducing tumor FN
    - Significant risk of increased tumor FP (false alarms)
    - Use only if lower weights don't reduce tumor FN enough
    """
    confusion_weight = 0.3


# ============================================================================
# USAGE EXAMPLES
# ============================================================================
"""
To train with different confusion weights:

# Conservative start (recommended):
nnUNetv2_train 305 2d 0 -tr nnUNetTrainerConfusion01

# Slightly more aggressive:
nnUNetv2_train 305 2d 0 -tr nnUNetTrainerConfusion015

# Balanced:
nnUNetv2_train 305 2d 0 -tr nnUNetTrainerConfusion02

# Aggressive (if conservative weights don't work):
nnUNetv2_train 305 2d 0 -tr nnUNetTrainerConfusion025

Each trainer creates its own results folder:
- nnUNetTrainerConfusion01__nnUNetPlans__2d/fold_0/
- nnUNetTrainerConfusion015__nnUNetPlans__2d/fold_0/
- nnUNetTrainerConfusion02__nnUNetPlans__2d/fold_0/
- etc.

To add a new weight (e.g., 0.12), just add:

class nnUNetTrainerConfusion012(nnUNetTrainerConfusionBase):
    confusion_weight = 0.12
"""
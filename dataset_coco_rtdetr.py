import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class CocoEvalDataset(Dataset):
    """
    CocoEvalDataset yields raw PIL images and target dictionaries with metadata
    (image_id, width, height) matching the notebook implementation.
    """
    def __init__(self, coco_gt_obj, img_dir, img_ids=None):
        """
        Initialize CocoEvalDataset.
        
        Args:
            coco_gt_obj (pycocotools.COCO): Pre-loaded COCO ground truth object.
            img_dir (str): Directory containing the images.
            img_ids (list of int, optional): Specific subset of image IDs to evaluate.
        """
        self.coco = coco_gt_obj
        if img_ids is not None:
            self.img_ids = [i for i in img_ids if i in coco_gt_obj.imgs]
        else:
            self.img_ids = coco_gt_obj.getImgIds()
            
        self.img_dir = img_dir
        self.img_info = coco_gt_obj.loadImgs(self.img_ids)
        
        # Create a mapping from image_id to file path
        self.id_to_path = {info['id']: os.path.join(img_dir, info['file_name'])
                           for info in self.img_info if 'file_name' in info}
        print(f"  CocoEvalDataset: Mapped {len(self.id_to_path)} image IDs to file paths.")
        
        # Verify paths exist
        missing_files = 0
        valid_img_ids = []
        for img_id in self.img_ids:
            path = self.id_to_path.get(img_id)
            if path and os.path.exists(path):
                valid_img_ids.append(img_id)
            else:
                missing_files += 1
        if missing_files > 0:
            print(f"  WARNING: Could not find image files for {missing_files} image IDs.")
        self.img_ids = valid_img_ids # Use only IDs with existing images
        print(f"  Using {len(self.img_ids)} valid image IDs for evaluation.")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = self.id_to_path[img_id] # Should exist based on init check
        try:
            image = Image.open(img_path).convert("RGB")
            # Target dict needed for post-processing size info and path
            target = {"image_id": img_id, "width": image.width, "height": image.height, "path": img_path}
            return image, target
        except Exception as e:
            print(f"Error loading/processing image {img_path} for id {img_id}: {e}")
            return None # Return None if image loading fails

def eval_collate_fn(batch):
    """
    Collate function matching the notebook implementation.
    """
    batch = [item for item in batch if item is not None]
    if not batch: 
        return None
    # Collate images and targets separately
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

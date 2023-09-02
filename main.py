from pathlib import Path 
import cv2 
import torch
import numpy as np
from tqdm import tqdm 
from segment_anything import sam_model_registry, SamPredictor
from utils import separate_objects, create_bbox_prompt, create_point_prompt, mask_postprocessing


# Set Hyperparams 
category = "wood"
data_path = Path("data/mvtec") / category
save_dir = Path("results") / category

sam_checkpoint = "weights/sam_h.pth"
model_type = "vit_h"
device = "cuda:0"

def main(img_preprocessing: bool = False
         ):
    img_paths = [p for p in data_path.glob("test/*/*.png") if "good" not in str(p)]
    mask_paths = [str(data_path / "ground_truth" / p.parent.name / p.stem) + "_mask.png" for p in img_paths]

    for img_p, mask_p in tqdm(zip(img_paths, mask_paths)):
        img = cv2.imread(str(img_p))
        if img_preprocessing:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.equalizeHist(img)
            img = np.stack([img] * 3, axis=-1)

        mask = cv2.imread(str(mask_p), 0)
        masks = separate_objects(mask)

        # Create prompt based on mask 
        bbox_prompt = create_bbox_prompt(masks)
        point_prompt, labels = create_point_prompt(masks)

        # Apply SAM 
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        predictor.set_image(img)

        if len(bbox_prompt) == 1:
            sam_masks, _, _ = predictor.predict(
                # point_coords=point_prompt,
                # point_labels=labels,
                box=bbox_prompt,
                multimask_output=False,
            )
            
            sam_mask = np.where(sam_masks[0] == True, 1, 0)
            sam_mask = mask_postprocessing(sam_mask)

        elif len(bbox_prompt) > 1: 
            bbox_prompt = torch.tensor(bbox_prompt,
                                       device=device)
            transformed_boxes = predictor.transform.apply_boxes_torch(bbox_prompt, img.shape[:2])
            sam_masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            sam_masks = sam_masks.detach().cpu().numpy()
            sam_mask = np.zeros(sam_masks.shape[-2:])
            
            for m in sam_masks:
                m = np.where(m[0] == True, 1, 0)
                m = mask_postprocessing(m)
                sam_mask += m

        save_path = save_dir / img_p.parent.name 
        save_path.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(save_path / img_p.name), img)
        cv2.imwrite(str(save_path / img_p.stem) + "_gt.png", mask)
        cv2.imwrite(str(save_path / img_p.stem) + "_sam.png", sam_mask)

if __name__ == "__main__":
    main(img_preprocessing=True)
from segment_anything import build_sam, SamPredictor 
predictor = SamPredictor(build_sam(checkpoint="$1"))
predictor.set_image($2)
masks, _, _ = predictor.predict($3)

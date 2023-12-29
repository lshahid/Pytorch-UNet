# This script runs the predict.py file for all images in a directory

# Image directory
img_dir=../output/imgs/

# Predicted mask directory
pred_mask_dir=../output/pred_masks/

# Loop over images
for img_file in "$img_dir"*
do
    # Get file names
    filename=$(basename "$img_file")
    pred_mask_file="$pred_mask_dir${filename%.jpg}.gif"
    
    # Run predict.py
    python3 ../predict.py -m ../checkpoints/checkpoint_epoch5.pth -i "$img_file" -o "$pred_mask_file"
done

# Image directory
img_dir=../output/imgs/

# Output directory
output_dir=../output/predicted_masks/

# Loop over images
for img_file in "$img_dir"*
do
    # Get file names
    filename=$(basename "$img_file")
    output_file="$output_dir${filename%.jpg}.gif"
    
    # Run predict.py
    python3 ../predict.py -m ../checkpoints/checkpoint_epoch5.pth -i "$img_file" -o "$output_file"
done

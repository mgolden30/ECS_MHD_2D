
# Input and output filenames
INPUT_PATTERN="%03d.png"
PALETTE="palette.png"
OUTPUT_GIF="output.gif"

# First pass: generate a color palette from the input images
ffmpeg -y -i "$INPUT_PATTERN" -vf "palettegen" "$PALETTE"

# Second pass: create the gif using the palette for better quality
ffmpeg -y -i "$INPUT_PATTERN" -i "$PALETTE" -lavfi "fps=15 [x]; [x][1:v] paletteuse" "$OUTPUT_GIF"

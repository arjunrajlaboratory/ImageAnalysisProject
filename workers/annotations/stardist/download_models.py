from stardist.models import StarDist2D

# Download and cache the pretrained 2D model
model_versatile = StarDist2D.from_pretrained('2D_versatile_fluo')

# Download and cache the pretrained 2D model for nuclei
model_nuclei = StarDist2D.from_pretrained('2D_versatile_he')

print("StarDist models downloaded and cached successfully.")
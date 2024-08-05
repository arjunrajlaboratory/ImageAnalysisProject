import sys
import traceback

def download_models():
    try:
        from stardist.models import StarDist2D

        # Download and cache the pretrained 2D model
        model_versatile = StarDist2D.from_pretrained('2D_versatile_fluo')

        # Download and cache the pretrained 2D model for nuclei
        model_nuclei = StarDist2D.from_pretrained('2D_versatile_he')

        print("StarDist models downloaded and cached successfully.")
    except ImportError as e:
        print(f"Error importing StarDist: {e}")
        print("Traceback:")
        traceback.print_exc()
        print("\nPlease check your environment and package versions.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    download_models()
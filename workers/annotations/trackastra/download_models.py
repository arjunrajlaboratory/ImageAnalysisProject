"""Pre-download the pretrained Trackastra model into the image.

Baking the weights into the Docker image avoids a network download on the
first run (NimbusImage launches a fresh container per job). Best-effort: if the
model host is unreachable at build time, the build continues and the model is
downloaded on first use instead.
"""

from trackastra.model import Trackastra

if __name__ == '__main__':
    # device="cpu" is used at build time since no GPU is present during build.
    Trackastra.from_pretrained("general_2d", device="cpu")
    print("Downloaded Trackastra general_2d model")

docker build ./workers/properties/lines/line_length_worker/ -t annotations/line_length:latest --build-arg BASE_IMAGE=ghcr.io/arjunrajlaboratory/base_x86_image:latest

# INTEL
docker build -f ./workers/annotations/connect_to_nearest/Dockerfile -t annotations/connect_to_nearest:latest ./workers/annotations/connect_to_nearest/
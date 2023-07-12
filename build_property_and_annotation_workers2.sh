docker build ./workers/annotations/random_square/ -t annotations/random_square:latest --build-arg BASE_IMAGE=ghcr.io/arjunrajlaboratory/base_m1_image:latest


docker build ./workers/properties/lines/line_length_worker/ -t annotations/line_length:latest --build-arg BASE_IMAGE=ghcr.io/arjunrajlaboratory/base_m1_image:latest
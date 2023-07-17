docker build ./workers/annotations/random_square/ -t annotations/random_square:latest --build-arg BASE_IMAGE=ghcr.io/arjunrajlaboratory/base_m1_image:latest


docker build ./workers/properties/lines/line_length_worker/ -t properties/line_length:latest --build-arg BASE_IMAGE=ghcr.io/arjunrajlaboratory/base_m1_image:latest

docker build ./workers/properties/points/point_to_nearest_point_distance/ -t properties/point_to_nearest_point_distance:latest --build-arg BASE_IMAGE=ghcr.io/arjunrajlaboratory/base_m1_image:latest

docker build ./workers/properties/points/point_to_nearest_connected_point_distance/ -t properties/point_to_nearest_connected_point_distance:latest --build-arg BASE_IMAGE=ghcr.io/arjunrajlaboratory/base_m1_image:latest
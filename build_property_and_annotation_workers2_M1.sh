docker build ./workers/annotations/random_square/ -t annotations/random_square:latest --build-arg BASE_IMAGE=ghcr.io/arjunrajlaboratory/base_m1_image:latest

docker build ./workers/annotations/random_point/ -t annotations/random_point:latest --build-arg BASE_IMAGE=ghcr.io/arjunrajlaboratory/base_m1_image:latest

docker build ./workers/annotations/connect_to_nearest/ -t annotations/connect_to_nearest:latest --build-arg BASE_IMAGE=ghcr.io/arjunrajlaboratory/base_m1_image:latest


docker build ./workers/properties/lines/line_length_worker/ -t properties/line_length:latest --build-arg BASE_IMAGE=ghcr.io/arjunrajlaboratory/base_m1_image:latest

docker build ./workers/properties/points/point_to_nearest_point_distance/ -t properties/point_to_nearest_point_distance:latest --build-arg BASE_IMAGE=ghcr.io/arjunrajlaboratory/base_m1_image:latest

docker build ./workers/properties/points/point_to_nearest_connected_point_distance/ -t properties/point_to_nearest_connected_point_distance:latest --build-arg BASE_IMAGE=ghcr.io/arjunrajlaboratory/base_m1_image:latest
############################################################################################################################################################################

# M1
docker build -f ./workers/annotations/connect_to_nearest/Dockerfile_M1 -t annotations/connect_to_nearest:latest ./workers/annotations/connect_to_nearest/

# INTEL
docker build -f ./workers/annotations/connect_to_nearest/Dockerfile -t annotations/connect_to_nearest:latest ./workers/annotations/connect_to_nearest/
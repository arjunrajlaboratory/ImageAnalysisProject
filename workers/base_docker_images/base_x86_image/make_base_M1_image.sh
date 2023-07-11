# This script builds the base docker image for x86 architecture.
# Run the following commented out command with your personal access token.
#export CR_PAT=[your personal access token]

echo $CR_PAT | docker login ghcr.io -u arjunrajlab --password-stdin

docker build -f Dockerfile -t ghcr.io/arjunrajlaboratory/base_x86_image:latest . --no-cache
docker push ghcr.io/arjunrajlaboratory/base_x86_image:latest


# This script builds the base docker image (test)
# Run the following commented out command with your personal access token.
#export CR_PAT=[your personal access token]

echo $CR_PAT | docker login ghcr.io -u arjunrajlab --password-stdin

docker build -f Dockerfile -t ghcr.io/arjunrajlaboratory/test_image:latest .
docker push ghcr.io/arjunrajlaboratory/test_image:latest


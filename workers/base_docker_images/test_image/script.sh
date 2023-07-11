export CR_PAT=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

echo $CR_PAT | docker login ghcr.io -u arjunrajlab --password-stdin

docker build -f Dockerfile -t ghcr.io/arjunrajlaboratory/test_image:latest .
docker push ghcr.io/arjunrajlaboratory/test_image:latest


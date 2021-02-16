#docker build -t tasm/environment -f docker/Dockerfile.environment .
docker build -t tasm/tasm -f docker/Dockerfile .
docker image prune -f

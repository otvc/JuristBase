sudo docker buildx build -t otvc/jurist-api .

sudo docker network create jurist-net

sudo docker run --name japi --net jurist-net otvc/jurist-api

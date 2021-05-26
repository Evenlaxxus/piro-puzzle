#docker build -t piro1_136280_136236 ./src
docker build -t piro1_136280_136236 - < piro1_136280_136236.src.tar.bz2
docker run --network none --mount type=bind,source=/home/zerkles/PiRO/piro-puzzle/src,target=/app/src,readonly piro1_136280_136236 /app/src/data/set0 6
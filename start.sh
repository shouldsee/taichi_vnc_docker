#MAIN=gcat_gromacs
MAIN=main
docker compose build --progress=auto && \
docker compose up -d && docker compose exec $MAIN bash 

## dependency

- docker compose plugin

## build

- `docker compose build`
- setup HTTPS_PROXY and HTTP_PROXY in `compose.yaml`

## usage 

- build and start the docker with `bash start.sh`
- connect to "http://localhost:9015"
- Start a terminal via: `startMenu -> Run -> lxterminal`
- test `python3 /data/threed.py`

## Performance

- `python3 ./python/tachi/examples/ggui_examples/mpm128_ggui.py` .  on tesla M40
  - noVNC web interface: 16.0 fps
  - ssh X11 forwarding:  0.60 fps

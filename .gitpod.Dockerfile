FROM gitpod/workspace-full-vnc

# Install custom tools, runtimes, etc.
# For example "bastet", a command-line tetris clone:
USER root
RUN apt-get update && apt-get install -y python-opengl
#
# More information: https://www.gitpod.io/docs/config-docker/

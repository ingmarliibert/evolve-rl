FROM gitpod/workspace-full-vnc

# Install custom tools, runtimes, etc.
# For example "bastet", a command-line tetris clone:
RUN apt-get install python-opengl
#
# More information: https://www.gitpod.io/docs/config-docker/

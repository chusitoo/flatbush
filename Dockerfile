FROM ubuntu:26.04

RUN DEBIAN_FRONTEND=noninteractive \
 && apt update -y \
 && apt upgrade -y \
 && apt install -y build-essential \
                   ca-certificates \
                   clang \
                   clang-format \
                   cmake \
                   cppcheck \
                   curl \
                   git \
                   msitools \
                   python3 \
                   tar \
                   unzip \
                   winbind \
                   wine64 \
                   zip \
 && apt clean all \
 && rm -rf /var/cache/apt

WORKDIR /opt

# install.sh creates ~/.wine; wait for wineserver to save its registry before the layer is committed.
RUN git clone https://github.com/mstorsjo/msvc-wine.git \
 && cd msvc-wine \
 && ./vsdownload.py --accept-license --dest /opt/msvc \
 && ./install.sh /opt/msvc \
 && /usr/lib/x86_64-linux-gnu/wine/wineserver -w

ENV PATH=/opt/msvc/bin/x64:$PATH

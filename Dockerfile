FROM ubuntu:24.04

RUN apt update -y \
 && apt install -y build-essential ca-certificates clang curl git tar unzip zip \
 && apt clean all \
 && rm -rf /var/cache/tdnf

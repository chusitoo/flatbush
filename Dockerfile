FROM mcr.microsoft.com/cbl-mariner/base/core:2.0

RUN tdnf update -y \
 && tdnf install -y build-essential ca-certificates clang++ curl git tar unzip zip \
 && tdnf clean all \
 && rm -rf /var/cache/tdnf

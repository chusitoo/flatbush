FROM mcr.microsoft.com/cbl-mariner/base/core:2.0

RUN tdnf update -y \
 && tdnf install -y build-essential tar \
 && tdnf clean all

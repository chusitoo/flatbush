#!/bin/bash

image=$(docker image ls --format "{{.Repository}}:{{.Tag}}" flatbush | head -1)

if [ "$image" == "" ] ; then
  image="flatbush:latest"
  docker build -t "$image" .
fi

docker run --rm -it -v "$PWD":/flatbush -w /flatbush --name flatbush "$image"

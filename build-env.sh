#!/bin/bash

docker run --rm -it -v $PWD:/flatbush -w /flatbush --name flatbush flatbush:latest

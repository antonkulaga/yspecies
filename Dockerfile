FROM quay.io/comp-bio-aging/base:latest

WORKDIR /opt/yspecies
ADD ./ /opt/yspecies
RUN micromamba env create -y -f /opt/yspecies/environment.yaml
ENV ENV_NAME="yspecies"
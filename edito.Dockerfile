FROM mambaorg/micromamba:1.5.6

USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER Makefile /tmp/Makefile
COPY --chown=$MAMBA_USER:$MAMBA_USER .condarc /tmp/.condarc
COPY --chown=$MAMBA_USER:$MAMBA_USER conda_environment_test.yaml /tmp/conda_environment_test.yaml
RUN CONDARC=.condarc micromamba install -y -n base make
ARG MAMBA_DOCKERFILE_ACTIVATE=1
COPY --chown=$MAMBA_USER:$MAMBA_USER pip.conf /tmp/pip.conf
RUN /usr/local/bin/_entrypoint.sh make create-test-environment
RUN ENV_NAME=oceanbench_test /usr/local/bin/_entrypoint.sh git init

ADD --chown=$MAMBA_USER:$MAMBA_USER oceanbench /oceanbench

CMD [ "make", "evaluate-challenger" ]

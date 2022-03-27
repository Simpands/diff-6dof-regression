FROM airlab404/dl:cuda10_pytorch_py36

RUN echo test
COPY requirements.yml /requirements.yml
RUN /bin/bash -c "conda update conda"

RUN /bin/bash -c ". activate py36 && conda env update -f=/requirements.yml"

CMD ["bash"]

WORKDIR /exp

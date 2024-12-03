FROM python:3.9

# Imposta la directory di lavoro all'interno del container
WORKDIR /app

# Copia i file dell'applicazione nel container
COPY . /app

# Installazione di Miniconda
RUN apt-get update && apt-get install -y wget bzip2 \
    && rm -rf /var/lib/apt/lists/* \
    && wget -qO miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh

# Aggiungi Conda al PATH
ENV PATH=/opt/conda/bin:$PATH

# Copia il file environment.yml nella directory di lavoro
COPY gazemdetr_env.yml .

# Crea un ambiente Conda a partire dal file environment.yml
RUN conda env create -n my_gaze-env -f gazemdetr_env.yml



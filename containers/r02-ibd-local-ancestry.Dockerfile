# The deployment records and verifies the resulting image ID before launch.
# This base is an already installed, authenticated project image.
FROM dnabr-m02-minor:20260922
RUN apt-get update && apt-get install --no-install-recommends -y bedtools=2.30.0+dfsg-3
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
RUN bedtools --version && python3 -c "import numpy; print(numpy.__version__)"

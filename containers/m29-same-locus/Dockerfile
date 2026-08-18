FROM dnabr-m28-lai-sim:0.1.1

# The base image already pins Python 3.11, numpy 2.4.6, tskit 1.0.3,
# msprime and stdpopsim.  M29 adds only the preregistered linear estimator.
USER root
RUN python3 -m pip install --no-cache-dir \
      scipy==1.16.3 \
      scikit-learn==1.7.2 \
    && python3 -c "import numpy, scipy, sklearn, tskit; assert sklearn.__version__ == '1.7.2'; assert tskit.__version__ == '1.0.3'"

USER runner

LABEL org.opencontainers.image.title="DNABR M29 same-locus DEV"
LABEL org.opencontainers.image.version="0.1.0"

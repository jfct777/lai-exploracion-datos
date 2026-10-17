FROM us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/dnabr-qc@sha256:3e9381d165b1e664e75093d885a62d8c180711c1c3f495324e8c08bdb5bd2af2 AS reader_tools

FROM us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-t0a@sha256:c03864a9ed0c56b00fd1a234daee2d17ddfa57d4c426628bd59cd9daf351ee99

# The production EventChromosome reader itself needs only NumPy. These pinned
# tools are copied from the already verified M16.5 image solely so the small
# artificial VCF fixture can be rebuilt inside this integration image without
# package downloads. plink2 is static; bcftools' Bookworm libraries are copied
# from the same immutable source image.
USER root
COPY --from=reader_tools /usr/local/bin/plink2 /usr/local/bin/plink2
COPY --from=reader_tools /usr/bin/bcftools /usr/bin/bcftools
COPY --from=reader_tools /lib/x86_64-linux-gnu/libhts.so.3 /lib/x86_64-linux-gnu/libhts.so.3
COPY --from=reader_tools /lib/x86_64-linux-gnu/libdeflate.so.0 /lib/x86_64-linux-gnu/libdeflate.so.0
COPY --from=reader_tools /lib/x86_64-linux-gnu/libhtscodecs.so.2 /lib/x86_64-linux-gnu/libhtscodecs.so.2
COPY --from=reader_tools /lib/x86_64-linux-gnu/liblzma.so.5 /lib/x86_64-linux-gnu/liblzma.so.5
COPY --from=reader_tools /lib/x86_64-linux-gnu/libbz2.so.1.0 /lib/x86_64-linux-gnu/libbz2.so.1.0
COPY --from=reader_tools /lib/x86_64-linux-gnu/libz.so.1 /lib/x86_64-linux-gnu/libz.so.1
COPY --from=reader_tools /usr/local/lib/python3.11/dist-packages/scipy /usr/local/lib/python3.11/site-packages/scipy
COPY --from=reader_tools /usr/local/lib/python3.11/dist-packages/scipy.libs /usr/local/lib/python3.11/site-packages/scipy.libs
COPY --from=reader_tools /usr/local/lib/python3.11/dist-packages/scipy-1.17.0.dist-info /usr/local/lib/python3.11/site-packages/scipy-1.17.0.dist-info
COPY --from=reader_tools /usr/local/lib/python3.11/dist-packages/sklearn /usr/local/lib/python3.11/site-packages/sklearn
COPY --from=reader_tools /usr/local/lib/python3.11/dist-packages/scikit_learn.libs /usr/local/lib/python3.11/site-packages/scikit_learn.libs
COPY --from=reader_tools /usr/local/lib/python3.11/dist-packages/scikit_learn-1.8.0.dist-info /usr/local/lib/python3.11/site-packages/scikit_learn-1.8.0.dist-info
COPY --from=reader_tools /usr/local/lib/python3.11/dist-packages/joblib /usr/local/lib/python3.11/site-packages/joblib
COPY --from=reader_tools /usr/local/lib/python3.11/dist-packages/joblib-1.5.2.dist-info /usr/local/lib/python3.11/site-packages/joblib-1.5.2.dist-info
COPY --from=reader_tools /usr/local/lib/python3.11/dist-packages/threadpoolctl.py /usr/local/lib/python3.11/site-packages/threadpoolctl.py
COPY --from=reader_tools /usr/local/lib/python3.11/dist-packages/threadpoolctl-3.6.0.dist-info /usr/local/lib/python3.11/site-packages/threadpoolctl-3.6.0.dist-info
RUN python3 -c "import numpy, scipy, sklearn, torch; assert numpy.__version__ == '2.4.6'; assert scipy.__version__ == '1.17.0'; assert sklearn.__version__ == '1.8.0'; assert torch.__version__ == '2.12.1+cpu'" \
    && bcftools --version | grep -F 'bcftools 1.16' \
    && plink2 --version | grep -F 'PLINK v2.0.0-a.7LM 64-bit Intel (10 Jan 2026)'

USER runner
WORKDIR /work

LABEL org.opencontainers.image.title="DNABR R02 real EventChromosome plus Torch CPU runtime" \
      org.opencontainers.image.version="torch2.12.1-numpy2.4.6-reader-tools-20261007t"

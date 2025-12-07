# Dockerfile
FROM nvcr.io/nvidia/tritonserver:23.10-py3

COPY entrypoint.sh /opt/triton/entrypoint.sh
RUN chmod +x /opt/triton/entrypoint.sh

ENTRYPOINT ["/opt/triton/entrypoint.sh"]


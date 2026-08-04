FROM python:3.10-trixie
RUN mkdir /tcs
WORKDIR /tcs
COPY . .
RUN pip install  --no-cache-dir -r requirements.txt
ENTRYPOINT ["/bin/bash"]
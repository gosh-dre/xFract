# IN DEV SERVER
# Download SSL certificates from GOSH
FROM gitlab.pangosh.nhs.uk:5050/dre-team/infrastructure/gosh_certs as certs

# Use the official Python 3.9.16 image as the base
FROM python:3.9.16

# Copy correct SSL certificates into container
COPY --from=certs /usr/local/share/ca-certificates/ /usr/local/share/ca-certificates/
COPY --from=certs /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
 
# Update and run the SSL certificates
RUN update-ca-certificates

ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt into container
COPY requirements.txt /app/

# Install dependencies in requirements.txt
RUN pip install -r /app/requirements.txt

# Copy the rest of project files into the container
COPY . /app/

# For API
RUN echo "Acquire { http::User-Agent \"Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/114.0\";};" > /etc/apt/apt.conf
RUN apt-get update
RUN ACCEPT_EULA=Y apt-get install -y curl gnupg
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/debian/11/prod.list | tee /etc/apt/sources.list.d/mssql-release.list
RUN apt-get update
RUN ACCEPT_EULA=Y DEBIAN_FRONTEND=noninteractive apt-get install -y \
    dnsutils \
    git \
    krb5-user \
    python3-full python3-pip \
    msodbcsql18 mssql-tools18 \
    vim

ADD krb5.conf .

# Set command to run either "main.py" (default) or pytest
CMD ["python", "main.py"]
 
# Copy in any changed files
RUN rm -f /app/main.py
COPY src/main.py /app/main.py
RUN rm -f /app/config/config.yaml
COPY config/config.yaml /app/config/config.yaml
RUN rm -f /app/prompts/prompt.json
COPY prompts/prompt.json /app/prompts/prompt.json
RUN rm -f /app/prompts/role_instructions.json
COPY prompts/role_instructions.json /app/prompts/role_instructions.json
COPY prompts/NLI_prompt.json /app/prompts/NLI_prompt.json
COPY components /app/components
RUN rm -f /app/input/Radiology_Synthetic_Data.csv
COPY tests/ /app/tests
COPY input/ /app/input
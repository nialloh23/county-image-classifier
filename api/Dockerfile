# Start from the latest Long Term Support (LTS) Ubuntu version
FROM ubuntu:18.04

# Install venv
RUN apt-get update && apt-get install python3-pip -y && pip3 install virtualenv

#Fix issue with openCV not working
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev && pip3 install opencv-python


# Create the working directory
RUN set -ex && mkdir /repo
WORKDIR /repo

# Copy only the relevant directories to the working diretory
COPY county_classifier/ ./county_classifier
COPY api/ ./api

# Install Python dependencies
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
RUN set -ex && pip3 install -r api/requirements.txt

# Run the web server
EXPOSE 8000
ENV PYTHONPATH /repo
CMD python3 /repo/api/app.py
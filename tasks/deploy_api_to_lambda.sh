#!/bin/bash

pip freeze -> api/requirements.txt
sed -i 's/tensorflow-gpu/tensorflow/' api/requirements.txt
cd api || exit 1
npm install
npx sls deploy -v
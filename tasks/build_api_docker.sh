#pipenv lock --requirements --keep-outdated > api/requirements.txt
pip freeze -> api/requirements.txt
sed -i 's/-gpu//g' api/requirements.txt
docker build -t county_classifier_api -f api/Dockerfile .
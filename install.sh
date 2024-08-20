#!/bin/bash

set -e
# set -x

PORT=5000
SSLLOC="/etc/nginx"
RUNDEFAULTIMPLEMETATION=false

Help() {
    echo "Install script for aimodel deployment."
    echo
    echo "Syntax: sudo ./install.sh [-h|p|s]"
    echo "h     Print this help."
    echo "p     Port where the flask application will be running."
    echo "r     Run default implementation (gpt4all, and Mistral model if GPU available)"
    echo "s     Location where key and certificate will be stored, defaults to $SSLLOC"
    echo
}

while getopts "hp:s:r" option; do
    case $option in
    h)
        Help
        exit
        ;;
    p)
        PORT=$OPTARG
        echo "Set port to $PORT"
        ;;
    s)
        SSLLOC=$OPTARG
        ;;
    r)
        RUNDEFAULTIMPLEMETATION=true
        ;;
    esac
done

# Location of the ssl certificate and key
SSLCERTIFICATE=$SSLLOC/aimodel_cert.pem
SSLKEY=$SSLLOC/aimodel_key.pem

# Force the user to run the script as sudo
if [[ $(id -u) -ne 0 ]]; then
    echo "Please run this file as sudo user."
    exit
fi

# Update the system
apt update

# Install python3.10 and pip
apt install -y python3.10-venv python3-pip

# Install pyopenssl to create certificates
sudo -u $SUDO_USER pip install pyopenssl

# Install nginx and gunicorn
apt install nginx
apt install -y gunicorn

# Create a self-signed certificate
openssl req -x509 -newkey rsa:4096 -nodes -out $SSLLOC/aimodel_cert.pem -keyout $SSLLOC/aimodel_key.pem -days 365 -subj="/C=CA/ST=Ontario/L=Ottawa/O=Carleton University/OU=RCS/CN=localhost"

# Change the owner of the certificate and key
sudo chown $SUDO_USER /etc/nginx/aimodel_*

# Create a new nginx configuration file
cat >/etc/nginx/sites-available/aimodel <<EOL
server {
    listen 80;
    listen 443 default_server ssl;
    ssl_certificate $SSLCERTIFICATE;
    ssl_certificate_key $SSLKEY;

    server_name _;

    location / {
        proxy_pass https://localhost:$PORT;
    }
} 
EOL

# Create a symbolic link to the sites-enabled directory
sudo ln -sf /etc/nginx/sites-available/aimodel /etc/nginx/sites-enabled/aimodel

# Check the configuration file
sudo nginx -t

# Restart the nginx service
sudo systemctl restart nginx

# Create a virtual environment
[ -d .venv ] &
echo ".venv already exists. skipping." | sudo -u $SUDO_USER python3 -m venv .venv

# Activate the virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo ".venv already active"
else
    source .venv/bin/activate
fi

# Upgrade pip
python -m pip install --upgrade pip

# Install the requirements
pip install -e .

# Install gunicorn
pip install gunicorn

# Run the default implementation (gpt4all, and Mistral model if GPU available)
if [ "$RUNDEFAULTIMPLEMETATION" == true ]; then
    GUNICORNCMD="gunicorn --certfile $SSLCERTIFICATE --keyfile $SSLKEY -b 0.0.0.0:$PORT --threads 8 --timeout 0  main:app"
    echo "$GUNICORNCMD"
    $GUNICORNCMD
fi

echo "Done!"

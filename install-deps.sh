#!/bin/bash

if [ -z "$VIRTUAL_ENV" ]
then
	echo "Please activate the virtual environment first"
	exit 3
fi

pip install -r requirements.txt --use-feature=2020-resolver

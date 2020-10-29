#!/bin/sh

mkdir -p data/speech_commands

if [ ! -e data/speech_commands/speech_commands_v0.02.tar.gz ]
then
	echo "Go to the below url and save the file into the data/speech_commands directory"
	echo ""
	echo "https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
	echo ""
fi

cd data/speech_commands \
	&& tar -zxvf data_speech_commands_v0.02.tar.gz \
	&& cd ../../

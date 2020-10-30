#!/bin/sh

random_string() {
  cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w "${1:-32}" | head -n 1
}

mkdir -p data/{speech_commands,temp}

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

echo "Importing custom words"
for i in custom_words/*
do
  word=$(basename "$i");
  echo "Importing $word"
  mkdir -p data/speech_commands/"$word"
  for f in custom_words/"$word"/**/*.wav
  do
    cp "$f" -v  data/speech_commands/"$word"/"$word"_"$(random_string 8)".wav
  done
done
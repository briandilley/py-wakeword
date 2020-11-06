#!/bin/sh

WORD="igor"

random_string() {
  cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w "${1:-32}" | head -n 1
}

mkdir -p data/speech_commands data/temp

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

for i in data/speech_commands/"$WORD"/*.wav
do
  f=$(basename "$i");
  if [ -z "${f##*copy_*}" ]
    then
      print "SKIPPING $f"
      continue;
  fi
  cp -v data/speech_commands/"$WORD"/$f data/speech_commands/"$WORD"/copy_1_"$f";
  cp -v data/speech_commands/"$WORD"/$f data/speech_commands/"$WORD"/copy_2_"$f";
  cp -v data/speech_commands/"$WORD"/$f data/speech_commands/"$WORD"/copy_3_"$f";
  cp -v data/speech_commands/"$WORD"/$f data/speech_commands/"$WORD"/copy_4_"$f";
  cp -v data/speech_commands/"$WORD"/$f data/speech_commands/"$WORD"/copy_5_"$f";
  cp -v data/speech_commands/"$WORD"/$f data/speech_commands/"$WORD"/copy_6_"$f";
  cp -v data/speech_commands/"$WORD"/$f data/speech_commands/"$WORD"/copy_7_"$f";
  cp -v data/speech_commands/"$WORD"/$f data/speech_commands/"$WORD"/copy_8_"$f";
  cp -v data/speech_commands/"$WORD"/$f data/speech_commands/"$WORD"/copy_9_"$f";
done
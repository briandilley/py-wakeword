# py-wakeword
Wakeword detection with TensorFlow

## env setup:
- ./venv.sh
- ./install-deps.sh
- ./prepare-datasets.sh

## training
- open train.py and set the appropriate `wake_word` at the top
- run train.py
- wait forever

## recording new shit

using the record script
`> python record.py --interactive --interactive_save_path ./data/temp --sample_rate 16000 --seconds 1`

using the record script in guided mode
> python record.py --guided --save_path ./data/temp --sample_rate 16000 --seconds 1 --recordings 50 --guided_pause 1

## cleaning up existing audio files

cleaning up audio files
`> ffmpeg -i file.audio -ar 16000 -t 00:00:01 file.audio.wav`

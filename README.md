# py-wakeword
Wakeword detection with TensorFlow

## env setup:
- ./venv.sh
- ./install-deps.sh
- ./download-datasets.sh

## recording new shit

using the record script
`> python record.py --interactive --interactive_save_path ./data/temp --sample_rate 16000 --seconds 1`

cleaning up audio files
`> ffmpeg -i file.audio -ar 16000 -t 00:00:01 file.audio.wav`

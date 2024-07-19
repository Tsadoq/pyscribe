import logging
import os

from transcriber.transcriber import Transcriber

logging.basicConfig(level=logging.INFO)


def test_run():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
    path = input('Enter the path to the audio file: ')
    min_speakers = int(input('Enter the minimum number of speakers: '))
    max_speakers = int(input('Enter the maximum number of speakers: '))
    transcriber = Transcriber()
    transcriber.transcribe_audio(
        audio_file=path,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )


if __name__ == '__main__':
    test_run()

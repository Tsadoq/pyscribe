import datetime
import logging
import os
import shutil
import uuid
from collections import defaultdict

import torch
from dotenv import find_dotenv, load_dotenv
from pyannote.audio import Pipeline
from pydub import AudioSegment
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from config.config import settings

load_dotenv(find_dotenv())


class Transcriber:

    def __init__(self, clean_up=True):
        self.whisper_pipeline = None
        self.whisper_model_id = None
        self.whisper_model = None
        self.base_segments_dir = None
        self.base_tmp_dir = None
        self.current_dir = None

        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"

        self.set_up_piannote()
        self.set_up_whisper()
        self.set_up(clean_up=clean_up)

    def set_up_whisper(self):
        if torch.cuda.is_available():
            whisper_device = 'cuda'
            logging.info(f'Using {torch.cuda.get_device_name()}')
        elif torch.backends.mps.is_available():
            logging.info('Using mps')
            whisper_device = 'mps'
        else:
            logging.info('Using cpu')
            whisper_device = 'cpu'
        self.whisper_model_id = "openai/whisper-large-v3"
        self.whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.whisper_model_id, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.whisper_model.to(whisper_device)
        processor = AutoProcessor.from_pretrained(self.whisper_model_id)
        self.whisper_pipeline = pipeline(
            task="automatic-speech-recognition",
            model=self.whisper_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            device=whisper_device,
        )

    def set_up_piannote(self):
        self.pipeline = Pipeline.from_pretrained(
            'pyannote/speaker-diarization-3.1',
            use_auth_token=settings.hf_token,
        )
        if torch.cuda.is_available():
            pyannote_device = torch.device('cuda')
            logging.info(f'Using {torch.cuda.get_device_name()}')
        elif torch.backends.mps.is_available():
            logging.info('Using mps')
            pyannote_device = torch.device('mps')
        else:
            logging.info('Using cpu')
            pyannote_device = torch.device('cpu')
        self.pipeline.to(pyannote_device)
        return pyannote_device

    def set_up(self, clean_up=True):
        logging.info('Setting up transcriber')
        self.current_dir = os.path.dirname(os.path.realpath(__file__))
        self.base_tmp_dir = f'{self.current_dir}/tmp_files'
        self.base_segments_dir = f'{self.current_dir}/segments'
        if clean_up:
            logging.info('Cleaning up tmp files')
            shutil.rmtree(path=self.base_tmp_dir, ignore_errors=True)
            shutil.rmtree(path=self.base_segments_dir, ignore_errors=True)
        os.makedirs(self.base_tmp_dir, exist_ok=True)
        os.makedirs(self.base_segments_dir, exist_ok=True)

    def transcribe_audio(self, audio_file, min_speakers=1, max_speakers=10):
        logging.info(f'Reading audio file {audio_file}')
        audio = AudioSegment.from_file(audio_file)
        tmp_audio_file = f'{self.base_tmp_dir}/{uuid.uuid4()}.wav'
        with open(tmp_audio_file, "wb") as f:
            audio.export(f, format="wav")
        logging.info(f'Exported audio file to {tmp_audio_file} for diarization')
        logging.info(f'Diarizing audio file {tmp_audio_file}')
        if max_speakers == 1 and min_speakers == 1:
            logging.info('Only one speaker detected. Transcribing the whole audio')
            diarization_files = [tmp_audio_file]
        else:
            if max_speakers < min_speakers:
                raise ValueError('max_speakers must be greater than min_speakers')
            elif max_speakers != min_speakers:
                diarization = self.pipeline(
                    tmp_audio_file,
                    num_speakers=min_speakers,
                )
            else:
                diarization = self.pipeline(
                    tmp_audio_file,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )

            speakers = self._clean_diarization_format(diarization)
            logging.info(f'Diarization completed. Found {len(speakers)} speakers')
            diarization_files = self._crate_diarization_files(audio, speakers)
            logging.info(f'Created {len(diarization_files)} diarization files')
            logging.info('Transcribing diarized files')
        results = self.whisper_transcribe(diarization_files)
        output_path = f'{self.base_tmp_dir}/{audio_file.split("/")[-1].split(".")[0]}_transcription.txt'
        logging.info(f'Writing transcription to {output_path}')
        with open(output_path, 'w+', encoding='utf-8') as output_file:
            for result in results:
                output_file.write(
                    f"{result['speaker']} ({result['start']} - {result['end']}): {result['transcription']}\n"
                )
        return results

    def whisper_transcribe(self, list_diarized_files):
        results = []
        if len(list_diarized_files) == 1:
            result = self.whisper_pipeline(list_diarized_files[0], generate_kwargs={"language": "italian"})
            results.append(
                {
                    'speaker': 'SPEAKER_1',
                    'start': 'start',
                    'end': 'end',
                    'transcription': result['text'],
                }
            )

        else:
            for idx, filename in enumerate(list_diarized_files):
                logging.info(f'Transcribing {filename}, {idx + 1}/{len(list_diarized_files)}')
                result = self.whisper_pipeline(filename, generate_kwargs={"language": "italian"})
                start = int(filename.split('/')[-1].split('_')[2])
                end = int(filename.split('/')[-1].split('_')[3].split('.')[0])
                results.append(
                    {
                        'speaker': filename.split('/')[-1].split('_')[0] + '_' + filename.split('/')[-1].split('_')[1],
                        'start': datetime.datetime.fromtimestamp(start / 1000.0).strftime('%M:%S'),
                        'end': datetime.datetime.fromtimestamp(end / 1000.0).strftime('%M:%S'),
                        'transcription': result['text'],
                    }
                )
            results.sort(key=lambda x: x['start'])
        return results

    def _crate_diarization_files(self, audio, speakers):
        list_diarized_files = []
        for speaker, segments in speakers.items():
            list_of_segments = self.unify_audio_intervals(segments, 1200)
            for segment in list_of_segments:
                start = segment['start']
                end = segment['end']
                audio_speaker = audio[start:end]
                filename = f'{self.base_segments_dir}/{speaker}_{start}_{end}.wav'
                list_diarized_files.append(filename)
                audio_speaker.export(
                    filename,
                    format='wav',
                )
        return list_diarized_files

    def unify_audio_intervals(self, list_of_dicts, threshold):
        merged_dicts = []
        i = 0
        while i < len(list_of_dicts):
            current_dict = list_of_dicts[i]
            while i + 1 < len(list_of_dicts) and list_of_dicts[i + 1]['start'] - current_dict['end'] <= threshold:
                next_dict = list_of_dicts[i + 1]
                current_dict['end'] = next_dict['end']
                i += 1
            merged_dicts.append(current_dict)
            i += 1
        return merged_dicts

    @staticmethod
    def _clean_diarization_format(diarization):
        speakers = defaultdict(list)
        for track in diarization.itertracks(yield_label=True):
            start = int(track[0].start * 1000)
            end = int(track[0].end * 1000)
            speaker = track[2]
            segment = track[1]
            speakers[speaker].append({'start': start, 'end': end, 'segment': segment})
        return speakers

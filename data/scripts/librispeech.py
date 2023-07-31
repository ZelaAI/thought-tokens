from datasets import Dataset
import datasets
import os

def _generate_examples(files, local_extracted_archive):
    """Generate examples from a LibriSpeech archive_path."""
    key = 0
    audio_data = {}
    transcripts = []
    for path in files:
        with open(path, "rb") as f:
            if path.endswith(".flac"):
                id_ = path.split("/")[-1][: -len(".flac")]
                audio_data[id_] = f.read()
            elif path.endswith(".trans.txt"):
                for line in f:1027
                
                    if line:
                        line = line.decode("utf-8").strip()
                        id_, transcript = line.split(" ", 1)
                        audio_file = f"{id_}.flac"
                        speaker_id, chapter_id = [int(el) for el in id_.split("-")[:2]]
                        audio_file = (
                            os.path.join(local_extracted_archive, audio_file)
                            if local_extracted_archive
                            else audio_file
                        )
                        transcripts.append(
                            {
                                "id": id_,
                                "speaker_id": speaker_id,
                                "chapter_id": chapter_id,
                                "file": audio_file,
                                "text": transcript,
                            }
                        )
            if audio_data and len(audio_data) == len(transcripts):
                for transcript in transcripts:
                    audio = {"path": transcript["file"], "bytes": audio_data[transcript["id"]]}

                    yield {"audio": audio, **transcript}
                    key += 1
                    
                    if key == 100: break
                audio_data = {}
                transcripts = []

def _dataset_generator():
    def files_generator():
        for root, dirs, filenames in os.walk("data.ignore/LibriSpeech"):
            for filename in filenames:
                yield os.path.join(root,filename)

    return _generate_examples(files_generator(), None)

def get_dataset():
    return Dataset.from_generator(_dataset_generator, features=datasets.Features(
        {
            "file": datasets.Value("string"),
            "audio": datasets.Audio(sampling_rate=16_000),
            "text": datasets.Value("string"),
            "speaker_id": datasets.Value("int64"),
            "chapter_id": datasets.Value("int64"),
            "id": datasets.Value("string"),
        }
    ))
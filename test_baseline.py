import torch
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from utils.params import get_params
from utils.dataset import AsrDataModule
from utils.functional import edit_distance

from pdb import set_trace


def main():
    # Set up Whisper
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=torch_dtype,
        device=device,
    )

    # Get parameters
    params = get_params()

    # Define model and data
    data_module = AsrDataModule(params)
    data_module.setup(stage='test')
    tokenizer = data_module.tokenizer
    dataset = data_module.test_data

    n_tokens, edit_dist = 0, 0
    for ii,sample in enumerate(tqdm(dataset)):
        x, y, _ = sample
        out = pipe(x, generate_kwargs={"language": "english"})
        y_hat = tokenizer(out['text'])
        if ii%500==0:
            print(f'TARGET: {tokenizer.decode(y)}')
            print(f'PREDICTED: {tokenizer.decode(y_hat)}')
        edit_dist += edit_distance(y, y_hat)
        n_tokens += len(y)
    wer = edit_dist/n_tokens
    print(f'WER: {wer}')
    

if __name__=='__main__':
    main()
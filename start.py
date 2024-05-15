import argparse
import os
import sys
import random

import torch
from PIL import Image
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, \
    KeywordsStoppingCriteria
from mplug_owl2.model.builder import load_pretrained_model
from transformers import TextStreamer

from pipeline import run_pipeline_by_question

mplugowl_model = None
MODEL_PATH = 'MAGAer13/mplug-owl2-llama2-7b'

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model():
    global mplugowl_model
    model_name = get_model_name_from_path(MODEL_PATH)
    mplugowl_model = load_pretrained_model(MODEL_PATH, None, model_name, load_8bit=False, load_4bit=False,
                                           device=device)


def vqa_task(image, row_data, choices=None):
    # return f'prediction, {image}, {row_data["question"]}'  # turn off model for pipeline testing

    if mplugowl_model is None:
        load_model()

    tokenizer, model, image_processor, context_len = mplugowl_model

    conv = conv_templates["mplug_owl2"].copy()
    roles = conv.roles

    img = Image.open(image).convert('RGB')
    max_edge = max(img.size)  # We recommand you to resize to squared image for BEST performance.
    img = img.resize((max_edge, max_edge))

    image_tensor = process_images([img], image_processor)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    list_of_choices = []

    if choices is None:
        question = row_data['question']
    else:
        question = row_data['question'] + '\n'
        for ii in range(len(choices)):
            list_of_choices.append({
                'symbol': chr(ii + 65),
                'choice': choices[ii]
            })
        for ii in range(len(list_of_choices)):
            question += f"{list_of_choices[ii]['symbol']}. {list_of_choices[ii]['choice']}\n"

        question += "Answer with the option's letter from the given choices directly."

    # print(question)

    with HiddenPrints():  # disable logging for faster inference
        inp = DEFAULT_IMAGE_TOKEN + question
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
            model.device)
        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        temperature = 0.7
        max_new_tokens = 512

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().replace('</s>', '')

    if choices is not None:
        return f'{outputs} | {[c["symbol"] + ". " + c["choice"] for c in list_of_choices]}'

    return outputs


def test_model():
    from pathlib import Path

    print('===== TEST MODEL =====')
    img = 'img/eiffel.jpg'
    assert Path(img).exists(), f'No image in {img}'
    row_data = {
        'question': 'In which country is this tower located?',
        'choices': ['France', 'Germany', 'Vietnam', 'China'],
        'choice_scores': [1, 0, 0, 0]
    }
    r = vqa_task(img, row_data, True)
    print(f'{img}, {row_data["question"]}, {r}')


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name', type=str, default='ReasonVQA', help='Valid input: ReasonVQA, VQAv2, OKVQA, GQA')
    parser.add_argument('--ds_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--img_dir', type=str, default='', help='Path to images')
    parser.add_argument('--output_dir_name', type=str, default='output', help='Path to output')
    parser.add_argument('--split', type=str, default='train', help='Set to "train" or "test"')
    parser.add_argument('--start_at', type=int, default=0, help='Index of the sample to start from')
    parser.add_argument('--limit', type=int, default=0, help='Max number of samples')
    parser.add_argument('--multichoice', action='store_true')
    args = parser.parse_args()

    if not args.img_dir:
        args.img_dir = args.ds_dir

    print(args)

    run_pipeline_by_question(vqa_task, args.ds_name, args.ds_dir, args.img_dir, args.output_dir_name, limit=args.limit,
                             start_at=args.start_at, split=args.split, multichoice=args.multichoice)


if __name__ == '__main__':
    # test_model()
    main()

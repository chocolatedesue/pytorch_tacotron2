import json
import logging

from traitlets import default
from tacotron2.text import text_to_sequence
import models
import torch
import argparse
import os
import numpy as np
from scipy.io.wavfile import write
from hifigan.models import Generator
from scipy.io.wavfile import write




def parse_args(parser):
    """
    Parse commandline arguments.
    """
    
    parser.add_argument('-t',"--text",type=str,help="the raw text to be synthesis")
    parser.add_argument('-i', '--input', type=str, default=None,
                        help='full path to the input text (phareses separated by new line)')
    parser.add_argument('-o', '--output', required=True,
                        help='output folder to save audio (file per phrase)')
    
    parser.add_argument('--tacotron2', type=str,required=True,
                        help='full path to the Tacotron2 model checkpoint file')
    parser.add_argument('--hifigan_path', type=str,required=True,
                        help='full path to the WaveGlow model checkpoint file')
    
    parser.add_argument('--hifigan_config',type=str,required=True, help = 'full path to the Tacotron2 model checkpoint file')
    
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('-c','--text_cleanner',type=str,default="japanese_tokenization_cleaners")
    
    
    
    '''
        Unsupported operation
   '''
    parser.add_argument('-s', '--sigma-infer', default=0.9, type=float)
    parser.add_argument('--suffix', type=str, default="", help="output filename suffix")
    parser.add_argument('-d', '--denoising-strength', default=0.01, type=float)
    
    
    
    

    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument('--cpu', action='store_true',
                    help='Run inference on CPU')
    
    
    '''
        Unsupported operation
    '''
    run_mode.add_argument('--fp16', action='store_true',
                        help='Run inference with mixed precision')

    parser.add_argument('--log-file', type=str, default='nvlog.json',
                        help='Filename for logging')
    parser.add_argument('--include-warmup', action='store_true',
                        help='Include warmup')
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')

    return parser



def load_tacotron2_model(model_name, parser, checkpoint, fp16_run, cpu_run, forward_is_infer=False):
    model_parser = models.model_parser(model_name, parser, add_help=False)
    model_args, _ = model_parser.parse_known_args()

    model_config = models.get_model_config(model_name, model_args)
    model = models.get_model(model_name, model_config, cpu_run=cpu_run,
                             forward_is_infer=forward_is_infer)

    if checkpoint is not None:
        if cpu_run:
            state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))['state_dict']
        else:
            state_dict = torch.load(checkpoint)['state_dict']
            
        model.load_state_dict(state_dict)

    model.eval()

    

    return model



def load_hifi_model(model_path,config_path, is_cpu):
    assert os.path.isfile(model_path) and os.path.isfile(config_path)
    device = torch.device("cpu" if is_cpu else "cuda")


    checkpoint_dict = torch.load(model_path, map_location=device)
    
    
    class AttrDict(dict):
            def __init__(self, *args, **kwargs):
                super(AttrDict, self).__init__(*args, **kwargs)
                self.__dict__ = self
                  
    with open (config_path,'r',encoding='utf-8') as f:
        data = f.read() 
    h = AttrDict(json.loads(data))
    torch.manual_seed(h.seed)
    
    audio_generator = Generator(h).to(device)
    audio_generator.load_state_dict(checkpoint_dict['generator'])
    audio_generator.eval()
    audio_generator.remove_weight_norm()

    return audio_generator

# taken from tacotron2/data_function.py:TextMelCollate.__call__
def pad_sequences(batch):
    # Right zero-pad all one-hot text sequences to max input length
    input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([len(x) for x in batch]),
        dim=0, descending=True)
    max_input_len = input_lengths[0]

    text_padded = torch.LongTensor(len(batch), max_input_len)
    text_padded.zero_()
    for i in range(len(ids_sorted_decreasing)):
        text = batch[ids_sorted_decreasing[i]]
        text_padded[i, :text.size(0)] = text

    return text_padded, input_lengths


def prepare_input_sequence(texts, cleanner_name,cpu_run=False):

    d = []
    for i,text in enumerate(texts):
        d.append(torch.IntTensor(
            text_to_sequence(text, [cleanner_name])[:]))

    text_padded, input_lengths = pad_sequences(d)
    if not cpu_run:
        text_padded = text_padded.cuda().long()
        input_lengths = input_lengths.cuda().long()
    else:
        text_padded = text_padded.long()
        input_lengths = input_lengths.long()

    return text_padded, input_lengths



def synthesis(model,generator,target_text,cleanner_name,device):
    text = target_text
    sample_rate = 22050
    
    
    sequence = np.array(text_to_sequence(text, [cleanner_name]))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).to(device).long()
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

    from scipy.io.wavfile import write
    with torch.no_grad():
        raw_audio = generator(mel_outputs.float())
        audio = raw_audio.squeeze()
        audio = audio * 32768.0
        audio = audio.to(device).numpy().astype('int16')
        write("audio_infer.wav", sample_rate, audio)
        
        
def main():
    """
    Launches text to speech (inference).
    Inference is executed on a single GPU or CPU.
    """
    parser = argparse.ArgumentParser(
        description='PyTorch Tacotron 2 Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()






    
    hifigan = load_hifi_model(args.hifigan_path,args.hifigan_config,args.cpu)
    tacotron2 = load_tacotron2_model('Tacotron2', parser, args.tacotron2,
                                     args.fp16, args.cpu, forward_is_infer=True)
    
    texts = []
    if args.input:
        f = open(args.input,'r')
        texts = f.readlines()
    elif args.text:
        texts = [args.text]
    else:
        print ("您未给出任何输入，程序退出")
        exit(0)

    sequences_padded, input_lengths = prepare_input_sequence(texts,args.text_cleanner, args.cpu)
    
    device = torch.device('cpu' if args.cpu else 'cuda')

    
    
    
    jitted_tacotron2 = torch.jit.script(tacotron2)
    with torch.no_grad():
        mel, mel_lengths, alignments = jitted_tacotron2(sequences_padded, input_lengths)
        
        
    with torch.no_grad():
        raw_audio = hifigan(mel.float())
        audio = raw_audio.squeeze()
        audio = audio * 32768.0
        audio = audio.cpu().numpy().astype('int16')
        write(os.path.join("output",'audio.wav'), 22050, audio)



if __name__ == '__main__':
    main()

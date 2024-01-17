import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='convert mmseg models to aux_head')
    parser.add_argument('input_path', help='the input checkpoint path')
    parser.add_argument('output_path', help='the output checkpoint path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    pretrained_model= torch.load(args.input_path)
    save_dir= args.output_path
    for k,v in pretrained_model.items():
        if k == "state_dict":
            pretrained_dict=v
    print("model_dict:")
    for k, v in pretrained_dict.items():
        print(k)
    for k in list(pretrained_dict.keys()):
        if k [0:14]== "auxiliary_head":
            pretrained_dict.pop(k)
        elif k [0:8]== "backbone":
            pretrained_dict.update({'teacher_backbone'+ k[8:]:pretrained_dict.pop(k)})
        elif k[0:11]=='decode_head':
            pretrained_dict.update({'teacher_head'+ k[11:]:pretrained_dict.pop(k)})
        else:
            pass
    print("PretrainModify:")
    for k, v in pretrained_dict.items():
        print(k)
    torch.save(pretrained_dict, save_dir)
import torch
import time
import argparse
from SCTNet import SCTNet_B,SCTNet_S


def parse_args():
    parser = argparse.ArgumentParser(description='Speed Measurement')
    
    parser.add_argument('--type', help='sctnet-b-seg100, sctnet-b-seg75,sctnet-b-seg50,sctnet-s-seg75,sctnet-s-seg50,sctnet-b-ade,sctnet-s-ade,sctnet-b-coco', default='sctnet-b-seg100', type=str)     
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True    
    device = torch.device('cuda')
    if args.type == 'sctnet-b-seg100':
        #63.01485739772993
        model = SCTNet_B()
        input = torch.randn(1, 3, 1024, 2048).cuda()
    elif args.type == 'sctnet-b-seg75' :
        #105.475097354823
        model = SCTNet_B()
        input = torch.randn(1, 3, 768, 1536).cuda()
    elif args.type == 'sctnet-b-seg50' :
        #144.54590529134427
        model = SCTNet_B()
        input = torch.randn(1, 3, 512, 1024).cuda()
    elif args.type == 'sctnet-s-seg75' :
        #150.68481278303713
        model = SCTNet_S()
        input = torch.randn(1, 3, 768, 1536).cuda()
    elif args.type == 'sctnet-s-seg50' :
        #155.75847537013587
        model = SCTNet_S()
        input = torch.randn(1, 3, 512, 1024).cuda()
    elif args.type == 'sctnet-b-ade' :
        #147.13018025501125
        model = SCTNet_B(num_classes=150)
        input = torch.randn(1, 3, 512, 512).cuda()
    elif args.type == 'sctnet-s-ade' :
        #160.58153924257374
        model = SCTNet_S(num_classes=150)
        input = torch.randn(1, 3, 512, 512).cuda()
    elif args.type == 'sctnet-b-coco' :
        #144.94364169474287
        model = SCTNet_B(num_classes=171)
        input = torch.randn(1, 3, 640, 640).cuda()
    model.eval()
    model.to(device)
    iterations = None
    with torch.no_grad():
        for _ in range(10):
            model(input)
    
        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)
    
        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)

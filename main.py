import os
import argparse
#from solver import Solver
#from data_loader import get_loader
from torch.backends import cudnn


def main(config):
    
    # 디렉터리 없으면 만들어!!!!
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    # dataset 옵션
    parser.add_argument("--dataroot",type=str,default="data",help="dataset 경로")
    parser.add_argument("--noise_kind",type=str,default=None,help="Gn이 거치는 noise 종류")
    parser.add_argument("--g_repeat_num",type=int,default=6,help="G-residual block 의 반복 수")
    parser.add_argument("--d_repeat_num",type=int,default=6,help="D-residual block 의 반복 수")
    
    # train or test
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
        
    # Trainig options
    parser.add_argument('--batch_size', type=int, default=32)    
    parser.add_argument('--g_lr', type=float, default=2e-4,)
    parser.add_argument('--d_lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.)
    parser.add_argument('--beta2', type=float, default=0.99)   
    parser.add_argument('--num_critic', type=int, default=1,help="G 훈련 대비 D 훈련 횟수")
    parser.add_argument('--channel_shuffle', action='store_true')
    parser.add_argument('--color_inversion', action='store_true')
    parser.add_argument('--blurvh', action='store_true')         
    parser.add_argument('--num_iterations', type=int, default=2000)   
    
    # Output options
    parser.add_argument('--visualize_interval', type=int, default=5000, help="시각화 간격")
    
    # Directories    
    parser.add_argument('--model_save_dir', type=str, default='./models',help="model의 가중치 저장 디렉터리")
    parser.add_argument('--model_load_dir',type=str,default='./models',help="사용 시에 가중치 저장되어 있는 디렉터리")
    
    
    config = parser.parse_args()
    
    
    print(config)
    main(config)
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    #common
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--compression_methods', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--budget', type=float, default=None)

    #oracle noargs
    #staic 
    parser.add_argument('--keep_high_dim', type=int, default=32)
    #frequency
    parser.add_argument("--min_k",type=int, default=1)
    parser.add_argument("--max_k",type=int, default=16)
    parser.add_argument("--threshold_ratio",type=float, default=0.8)
    parser.add_argument("--dataset_name",type=str, default="")
    parser.add_argument("--cal_dataset",type=str, default="qasper")
    parser.add_argument("--layer_control",type=int, default=-1)
    #quest
    parser.add_argument("--chunk_size",type=int, default=16)
    parser.add_argument("--evaluate_num",type=int, default=8)
    #snapkv
    
    
    
    return parser.parse_args(args)

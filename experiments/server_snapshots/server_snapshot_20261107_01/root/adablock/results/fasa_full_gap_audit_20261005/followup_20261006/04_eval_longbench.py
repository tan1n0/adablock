def eval_longbench(model,tokenizer,model_name,datasets,output_dir,evaluate_num=-1,layer_control=-1):
    rsts = {}
    max_length = model2maxlen[model_name]
    # predict on each dataset
    for dataset in datasets:
        file_path = f"{output_dir}/{dataset}_preds.jsonl"
        data = read_jsonl(f"../longbench/{dataset}.jsonl") if evaluate_num==-1 else read_jsonl(f"../longbench/{dataset}.jsonl")[:evaluate_num]
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        if os.path.exists(file_path):
            os.remove(file_path)
        rst = get_pred_single_gpu(data_all, max_length, max_gen, prompt_format, dataset, model_name, file_path,tokenizer,model)
        rsts[dataset] = rst
    scores = dict()
    print("Evaluating on:", datasets)
    print(f"{model_name}")
    for dataset, all_data in rsts.items():
        predictions, answers, lengths = [], [], []
        for data in all_data:
            predictions.append(data["pred"])
            answers.append(data["answers"])
            all_classes = data["all_classes"]
            if "length" in data:
                lengths.append(data["length"])
        score = scorer(dataset, predictions, answers, all_classes)
        scores[f"{dataset}_{layer_control}"] = score
        print(f"{dataset}_layer-{layer_control}: {scores[f'{dataset}_{layer_control}']}")
    out_path = f"{output_dir}/{model_name}_result.json"
    with open(out_path, "a") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

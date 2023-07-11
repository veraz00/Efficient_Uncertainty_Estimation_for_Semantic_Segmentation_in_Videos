def validate_video(args, model, split, labeled_index=None, verbose=False):
    num_classes= args.n_classes
    frane_names = json.load(open(args.root + 'data_split.json', 'r'))[split]['frames']
    frame_names = list(np.concatenate(frame_names))
    labeled_image_names = json.load(open(args.root + 'data_split.json', ;r))[split]['labeled_frames']

    model.eval()

    if args.model.is_bayesian:
        model.apply(set_dropout)
    
    if args.flow = 'DF':
        DF = cv2.optflow.createOptFlow_DeepFlow()
    elif args.flow == 'flownet2':
        flownet2 = cv2.optflow.readOpticalFlow('flownet2')
        path = args.flow_model_path 
        if os.path.exists(path) == False:
            raise ValueError('flow model path should not be empty')
        pretrained_dict = torch.load(path)['stat_dict']
        model_dict = flownet2.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        flownet2.load_state_dict(model_dict)
        flownet2.eval()
        flownet2.cuda()

    threshold = args.error_thres
    alpha_normal = args.alpha_normal
    alpha_error = args.alpha_error

    prev_frame = None 
    gts, preds, uncts = [], [], []
    uncts_r, uncts_e, uncts_v, uncts_b = [], [], [], []
    video_name = ''
    inference_time = 0

    for i, frame_name in enumerate(frame_names):
        torch.cuda.synchronize()
        t1 = time.time()
        img = cv2.imread(frame_name)
    
    output = F.softmax(model(images))
    

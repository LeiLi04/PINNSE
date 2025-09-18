def main():
    """
    PINNSE 训练/测试主入口，支持自动数据集分割、目录检测、实验日志保存。
    ---
    全流程修正版，无任何 model.rnn 依赖，所有超参数均从配置字典获取。
    """
    # ============================ Colab/脚本专用参数 ============================
    mode = "train"  # 或 "test"
    model_type = "gru"  # 或 "lstm", "rnn"
    dataset_type = "LorenzSSM"
    # trajectories_m_3_n_3_LorenzSSM_data_T_100_N_1000_r2_0dB_nu_-10dB
    # trajectories_m_3_n_3_LorenzSSM_data_T_100_N_1000_r2_10dB_nu_0dB
    # trajectories_m_3_n_3_LorenzSSM_data_T_100_N_1000_r2_-10dB_nu_-20dB
    # trajectories_m_3_n_3_LorenzSSM_data_T_100_N_1000_r2_20dB_nu_10dB
    # trajectories_m_3_n_3_LorenzSSM_data_T_100_N_1000_r2_-20dB_nu_-30dB

    datafile = "./data/trajectories_m_3_n_3_LorenzSSM_data_T_100_N_1000_r2_-20dB_nu_-30dB"
    splits_file = "trajectories_m_3_n_3_LorenzSSM_data_T_100_N_1000_r2_-20dB_nu_-30dB_split.pkl"
    model_file_saved = "./models/lorenz_model.pt"  # 仅test需要
    '''
    # ==========================================================================
    # STEP 02: 数据参数与设备设定
    # ==========================================================================
    print("datafile: {}".format(datafile))
    #   datafile = "./data/trajectories_m_3_n_3_LorenzSSM_data_T_200_N_1000.pkl"
    #   datafile.split('/') → [".", "data", "trajectories_m_3_n_3_LorenzSSM_data_T_200_N_1000.pkl"]
    #   datafile.split('/')[-1] → "trajectories_m_3_n_3_LorenzSSM_data_T_200_N_1000.pkl"
    print(datafile.split('/')[-1])
    # 从文件名解析参数
    from parse import parse
    # Add the missing '.pkl' extension for parsing
    parsed_datafile = datafile.split('/')[-1] + ".pkl"
    _, n_states, n_obs, _, T, N_samples, inverse_r2_dB, nu_dB = parse(
        "{}_m_{:d}_n_{:d}_{}_data_T_{:d}_N_{:d}_r2_{:f}dB_nu_{:f}dB.pkl", parsed_datafile
    )
    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")
    print("Device Used:{}".format(device))
    '''
    # ======================================================================
    # STEP 02: 数据参数与设备设定
    # ======================================================================
    print("datafile: {}".format(datafile))
    print(datafile.split('/')[-1])

    from parse import parse
    import os

    # === 修改：仅用 endswith 判断是否缺少 .pkl；避免 splitext 把 ".0dB" 误判为扩展名
    if not datafile.endswith(".pkl"):
        datafile = datafile + ".pkl"              # 自动补 .pkl
    # 同步得到规范化后的 basename
    basename = os.path.basename(datafile)

    # === 修改：按 .pkl 结尾的模板解析（与数据集生成规则一致）
    pattern = "{}_m_{:d}_n_{:d}_{}_data_T_{:d}_N_{:d}_r2_{}dB_nu_{}dB.pkl"
    parsed = parse(pattern, basename)
    if parsed is None:
        # === 修改：清晰报错，提示期望的文件名模板
        raise ValueError(
            f"Filename format mismatch: {basename}\n"
            f"Expected pattern: '{pattern}'"
        )

    # === 修改：从 parsed.fixed 安全解包
    _, n_states, n_obs, _, T, N_samples, inverse_r2_dB, nu_dB = parsed.fixed

    inverse_r2_dB = to_float(inverse_r2_dB)   # e.g. "0" → 0.0, "-10" → -10.0, "40.0" → 40.0
    nu_dB         = to_float(nu_dB)

    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print("Device Used:{}".format(device))

    # ==========================================================================
    # STEP 03: 获取参数字典
    # ==========================================================================
    # from parameters import get_parameters, get_H_DANSE
    ssm_parameters_dict, est_parameters_dict = get_parameters(
        N=N_samples, T=T, n_states=n_states, n_obs=n_obs,
        inverse_r2_dB=inverse_r2_dB, nu_dB=nu_dB, device=device
    )
    batch_size = est_parameters_dict["pinnse"]["batch_size"]
    estimator_options = est_parameters_dict["pinnse"]
    estimator_options['H'] = get_H_DANSE(type_=dataset_type, n_states=n_states, n_obs=n_obs)

    # ==========================================================================
    # STEP 04: 加载/生成数据集
    # ==========================================================================
    # from utils.utils import load_saved_dataset, Series_Dataset, obtain_tr_val_test_idx, create_splits_file_name, load_splits_file, get_dataloaders, check_if_dir_or_file_exists
    import pickle as pkl
    import os
    if not os.path.isfile(datafile):
        print("Dataset is not present, run 'generate_data.py / run_generate_data.sh' to create the dataset")
        return
    else:
        print("Dataset already present!")
        Z_XY = load_saved_dataset(filename=datafile)
    Z_XY_dataset = Series_Dataset(Z_XY_dict=Z_XY)

    # ==========================================================================
    # STEP 05: 数据集分割与保存
    # ==========================================================================
    if not os.path.isfile(splits_file):
        tr_indices, val_indices, test_indices = obtain_tr_val_test_idx(
            dataset=Z_XY_dataset, tr_to_test_split=0.9, tr_to_val_split=0.833)
        print(len(tr_indices), len(val_indices), len(test_indices))
        splits = {"train": tr_indices, "val": val_indices, "test": test_indices}
        splits_file_name = create_splits_file_name(dataset_filename=datafile, splits_filename=splits_file)
        print("Creating split file at:{}".format(splits_file_name))
        with open(splits_file_name, 'wb') as handle:
            pkl.dump(splits, handle, protocol=pkl.HIGHEST_PROTOCOL)
    else:
        print("Loading the splits file from {}".format(splits_file))
        splits = load_splits_file(splits_filename=splits_file)
        tr_indices, val_indices, test_indices = splits["train"], splits["val"], splits["test"]

    # ==========================================================================
    # STEP 06: 构建数据加载器
    # ==========================================================================
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset=Z_XY_dataset, batch_size=batch_size,
        tr_indices=tr_indices, val_indices=val_indices, test_indices=test_indices
    )
    print("No. of training, validation and testing batches: {}, {}, {}".format(
        len(train_loader), len(val_loader), len(test_loader)))

    # ==========================================================================
    # STEP 07: 日志与模型目录检查
    # ==========================================================================
    '''
    ./log/
    └── LorenzSSM_pinnse_gru_m_3_n_3_T_200_N_1000_40.0dB_0.0dB/
        ├── training.log
        └── testing.log

    ./models/
    └── LorenzSSM_pinnse_gru_m_3_n_3_T_200_N_1000_40.0dB_0.0dB/
        ├── pinnse_gru_ckpt_epoch_50.pt
        ├── pinnse_gru_ckpt_epoch_100.pt
        └── pinnse_gru_ckpt_epoch_200_best.pt

    '''
    logfile_path = "./log/"
    modelfile_path = "./models/"
    main_exp_name = "{}_pinnse_{}_m_{}_n_{}_T_{}_N_{}_{}dB_{}dB".format(
        dataset_type, model_type, n_states, n_obs, T, N_samples, inverse_r2_dB, nu_dB
    )
    tr_log_file_name = "training.log"
    te_log_file_name = "testing.log"
    flag_log_dir, flag_log_file = check_if_dir_or_file_exists(
        os.path.join(logfile_path, main_exp_name), file_name=tr_log_file_name)
    print("Is log-directory present:? - {}".format(flag_log_dir))
    print("Is log-file present:? - {}".format(flag_log_file))
    flag_models_dir, _ = check_if_dir_or_file_exists(
        os.path.join(modelfile_path, main_exp_name), file_name=None)
    print("Is model-directory present:? - {}".format(flag_models_dir))
    tr_logfile_name_with_path = os.path.join(os.path.join(logfile_path, main_exp_name), tr_log_file_name)
    te_logfile_name_with_path = os.path.join(os.path.join(logfile_path, main_exp_name), te_log_file_name)
    if flag_log_dir == False:
        print("Creating {}".format(os.path.join(logfile_path, main_exp_name)))
        os.makedirs(os.path.join(logfile_path, main_exp_name), exist_ok=True)
    if flag_models_dir == False:
        print("Creating {}".format(os.path.join(modelfile_path, main_exp_name)))
        os.makedirs(os.path.join(modelfile_path, main_exp_name), exist_ok=True)
    modelfile_path = os.path.join(modelfile_path, main_exp_name)
    '''
    # 新版
    # ==========================================================================
    # STEP 08: 训练与测试流程分支
    # ==========================================================================
    if mode.lower() == "train":
        # ==== 1. DANSE参数解析 ====
        danse_cfg = est_parameters_dict['danse']
        n_states = danse_cfg['n_states']
        n_obs = danse_cfg['n_obs']
        H = danse_cfg['H'] if danse_cfg['H'] is not None else np.eye(n_obs, n_states)
        C_w = danse_cfg['C_w']
        mu_w = danse_cfg['mu_w']
        batch_size = danse_cfg['batch_size']
        rnn_type = danse_cfg['rnn_type']
        rnn_params_dict = danse_cfg['rnn_params_dict']
        hidden_dim = rnn_params_dict[rnn_type]['n_hidden']
        n_layers   = rnn_params_dict[rnn_type]['n_layers']
        num_epochs = rnn_params_dict[rnn_type]['num_epochs']
        lr = rnn_params_dict[rnn_type]['lr']

        H_torch    = torch.tensor(H,    dtype=torch.float32, device=device)
        C_w_torch  = torch.tensor(C_w,  dtype=torch.float32, device=device)
        mu_w_torch = torch.tensor(mu_w, dtype=torch.float32, device=device)

        # ==== 2. 初始化子网 ====
        # from src.danse import PriorRNN, DeltaKNet, DANSEModel, train_danse # 路径按实际项目结构调整
        prior_net = PriorRNN(
            input_size=n_obs,
            hidden_size=hidden_dim,
            output_size=n_states,
            n_layers=n_layers,
            model_type=rnn_type
        ).to(device)

        deltaK_net = DeltaKNet(
            n_state=n_states,
            n_meas=n_obs,
            hidden=128,
            alpha=0.1
        ).to(device)

        model_danse = DANSEModel(
            prior_net=prior_net,
            deltaK_net=deltaK_net,
            H=H_torch,
            C_w=C_w_torch,
            mu_w=mu_w_torch,
            n_states=n_states,
            n_obs=n_obs,
            device=device
        ).to(device)

        tr_verbose = True
        save_chkpoints = "some"
        tr_losses, val_losses, _, _, _ = train_danse(
            model=model_danse,
            train_loader=train_loader,
            val_loader=val_loader,
            options=estimator_options,
            nepochs=num_epochs,    # <-- 用配置直接传递
            logfile_path=tr_logfile_name_with_path,
            modelfile_path=modelfile_path,
            save_chkpoints=save_chkpoints,
            device=device,
            tr_verbose=tr_verbose
        )
        # 保存loss
        from utils.utils import NDArrayEncoder
        with open(os.path.join(os.path.join(logfile_path, main_exp_name),
            'danse_{}_losses_eps{}.json'.format(estimator_options['rnn_type'], num_epochs)), 'w') as f:
            f.write(json.dumps({"tr_losses": tr_losses, "val_losses": val_losses}, cls=NDArrayEncoder, indent=2))

    elif mode.lower() == "test":
        from src.danse import test_danse
        te_loss = test_danse(
            model=model_danse,
            test_loader=test_loader,
            device=device,
            model_file=model_file_saved,
            test_logfile_path=te_logfile_name_with_path
        )
    return None
    '''
    # ==========================================================================
    # STEP 08: 训练与测试流程分支（旧版 DANSE 流程）
    # ==========================================================================
    if mode.lower() == "train":
        # === 改动：不再构建 PriorRNN/DeltaKNet/DANSEModel；直接用旧版 DANSE ===
        # 如果旧版类和函数就在当前文件/模块，请按你的实际路径导入：
        # from src.danse_legacy import DANSE, train_danse
        # 或者如果类名就在本文件上方定义：直接使用 DANSE / train_danse
        # from src.danse import DANSE, train_danse  # ← 按你的项目路径调整

        # 旧版 DANSE 直接吃 estimator_options（里边含 H, C_w, mu_w, mu_x0, C_x0 等）
        estimator_options = est_parameters_dict["pinnse"]
        estimator_options["H"] = get_H_DANSE(type_=dataset_type, n_states=n_states, n_obs=n_obs)

        # 实例化 PINNSE
        model_pinnse = PINNSE(**estimator_options).to(device)

        # 从配置里取训练轮数
        rnn_type = estimator_options["rnn_type"]
        num_epochs = estimator_options["rnn_params_dict"][rnn_type]["num_epochs"]

        # 开始训练（旧版 train_pinnse 的签名是：model, options, train_loader, val_loader, nepochs, ...）
        tr_verbose = True
        save_chkpoints = "some"
        tr_losses, val_losses, _, _, _ = train_pinnse(
            model=model_pinnse,
            options=estimator_options,
            train_loader=train_loader,
            val_loader=val_loader,
            nepochs=num_epochs,
            logfile_path=tr_logfile_name_with_path,
            modelfile_path=modelfile_path,
            save_chkpoints=save_chkpoints,
            device=device,
            tr_verbose=tr_verbose
        )

        # 保存 loss（保持原逻辑）
        # from utils.utils import NDArrayEncoder
        with open(os.path.join(logfile_path, main_exp_name,
                f'pinnse_{estimator_options["rnn_type"]}_losses_eps{num_epochs}.json'), 'w') as f:
            import json
            f.write(json.dumps({"tr_losses": tr_losses, "val_losses": val_losses}, cls=NDArrayEncoder, indent=2))

    elif mode.lower() == "test":
        # === 改动：旧版 test_pinnse 的签名是 (test_loader, options, device, model_file, test_logfile_path)
        from src.pinnse import test_danse  # ← 按你的项目路径调整

        te_loss = test_pinnse(
            test_loader=test_loader,
            options=estimator_options,          # === 改动：传 options（旧版 test 会内部构建 PINNSE 并 load 权重）
            device=device,
            model_file=model_file_saved,
            test_logfile_path=te_logfile_name_with_path
        )

if __name__ == "__main__":
    main()
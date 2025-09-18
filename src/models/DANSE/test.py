def test_danse(test_loader, options, device, model_file=None, test_logfile_path=None):
    """
    评估 DANSE 模型在测试集上的性能

    Args:
        test_loader (DataLoader): 测试集数据加载器
        options (dict): DANSE模型参数配置
        device (str): 运算设备
        model_file (str): 已保存模型文件路径
        test_logfile_path (str): 测试日志文件保存路径

    Returns:
        test_mse_loss (float): 测试集均方误差（MSE）

    Tensor Dimensions:
        te_Y_batch:         [B, T, n_obs] 观测序列
        te_X_batch:         [B, T, n_states] 真实状态序列
        te_mu_X_filtered_batch: [B, T, n_states] 预测的后验均值序列

    处理流程:
        1) 加载模型参数，切换到评估模式
        2) 逐 batch 计算对数似然和 MSE，统计整体性能
        3) 日志写入结果，支持单样本预测可视化

    工程细节:
        - 支持高效无梯度推断
        - 可方便扩展用于序列可视化和对比分析
    """
    # ==========================================================================
    # STEP 01: 加载模型与日志路径
    # ==========================================================================
    test_loss_epoch_sum = 0.0
    te_log_pY_epoch_sum = 0.0
    print("################ Evaluation Begins ################ \n")

    model = DANSE(**options)
    # model.load_state_dict(torch.load(model_file))
    # TODO[load]: 保证加载到正确设备（CPU/GPU），避免 device mismatch
    state = torch.load(model_file, map_location=device)   # << 改：加 map_location
    model.load_state_dict(state)
    criterion = nn.MSELoss()
    model = push_model(nets=model, device=device)
    model.eval()
    if test_logfile_path is None:
        test_log = "./log/test_danse.log"
    else:
        test_log = test_logfile_path

    X_ref = None
    X_hat_ref = None

    # ==========================================================================
    # STEP 02: 推断与评估
    # ==========================================================================
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            te_Y_batch, te_X_batch = data
            Y_test_batch = te_Y_batch.float().to(device)
            te_mu_X_predictions_batch, te_var_X_predictions_batch, te_mu_X_filtered_batch, te_var_X_filtered_batch = model.compute_predictions(Y_test_batch)
            log_pY_test_batch = -model.forward(Y_test_batch)
            # test_mse_loss_batch = criterion(te_X_batch, te_mu_X_filtered_batch)
            # TODO[metric-consistency]: 和验证集保持一致、避免冷启动误差，建议跳过 t=0
            # 同时要把 target 放到同一设备，避免 device mismatch
            # （如果你的 te_mu_X_filtered_batch 包含 T 个时间步，就对二者都做 [:,1:,:]）
            test_mse_loss_batch = criterion(
                te_X_batch[:, 1:, :].to(device),          # << 改：对齐到设备 + 跳过 t=0
                te_mu_X_filtered_batch[:, 1:, :]          # << 改：与 target 同样切片
            )
            test_loss_epoch_sum += test_mse_loss_batch.item()
            te_log_pY_epoch_sum += log_pY_test_batch.item()
        X_ref = te_X_batch[-1] # （B,T,D）
        X_hat_ref = te_mu_X_filtered_batch[-1]

    # ==========================================================================
    # STEP 03: 计算并记录最终指标
    # ==========================================================================
    test_mse_loss = test_loss_epoch_sum / len(test_loader)
    test_NLL_loss = te_log_pY_epoch_sum / len(test_loader)

    print('Test NLL loss: {:.3f}, Test MSE loss: {:.3f} using weights from file: {} %'.format(
        test_NLL_loss, test_mse_loss, model_file))

    # with open(test_log, "a") as logfile_test:
    #     logfile_test.write('Test NLL loss: {:.3f}, Test MSE loss: {:.3f} using weights from file: {}'.format(
    #         test_NLL_loss, test_mse_loss, model_file))
    # TODO[logging]: 建议加上换行，避免多次追加时挤在一行
    with open(test_log, "a") as logfile_test:
        logfile_test.write('Test NLL loss: {:.3f}, Test MSE loss: {:.3f} using weights from file: {}\n'.format(
            test_NLL_loss, test_mse_loss, model_file))

    # 可视化留作后续补充
    #plot_state_trajectory(X=X_ref, X_est=X_hat_ref)
    #plot_state_trajectory_axes(X=X_ref, X_est=X_hat_ref)

    return test_mse_loss
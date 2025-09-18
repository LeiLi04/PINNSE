# ====== train_danse函数，必须顶格（与DANSE同级，不在DANSE类内部！） ======
def train_pinnse(
    model, options, train_loader, val_loader, nepochs,
    logfile_path, modelfile_path, save_chkpoints,
    device='cpu', tr_verbose=False
):
    """
    训练 DANSE 模型（带有早停和模型保存机制）
    # ...docstring略...
    """
    # ==========================================================================
    # STEP 01: 初始化与日志输出
    # ==========================================================================
    model = push_model(nets=model, device=device)
    total_num_params, total_num_trainable_params = count_params(model)
    model.train()
    mse_criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=model.rnn.lr)
    # optimizer = torch.optim.Adam(model_danse.parameters(), lr=rnn_params_dict[rnn_type]['lr'])
    # +++++++修改过++++++
    lr = options['rnn_params_dict'][model.rnn_type]['lr']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # StepLR = 每隔固定步数 (epoch) 就把学习率按一个因子缩小。
    #  每次衰减时：
    #   new_lr = old_lr × γ
    #   这里 gamma=0.9 → 每次衰减后学习率变为原来的 90%。
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=nepochs//6, gamma=0.9)
    tr_losses = []
    val_losses = []
    # -----------------File--------------------------------
    if modelfile_path is None:
        model_filepath = "./models/"
    else:
        model_filepath = modelfile_path

    if save_chkpoints == "all" or save_chkpoints == "some":
        if logfile_path is None:
            training_logfile = "./log/pinnse_{}.log".format(model.rnn_type)
        else:
            training_logfile = logfile_path
    elif save_chkpoints is None:
        if logfile_path is None:
            training_logfile = "./log/gs_training_pinnse_{}.log".format(model.rnn_type)
        else:
            training_logfile = logfile_path
    # -----------------File_End--------------------------------
    # -----------------初始化----------------------------------
    # 1. EarlyStopping相关
    patience = 0
    num_patience = 3
    min_delta = options['rnn_params_dict'][model.rnn_type]["min_delta"]  # 容忍度设定，控制早停, 子dictionary
    check_patience = False
    # 2. 最优模型追踪
    best_val_loss = np.inf
    tr_loss_for_best_val_loss = np.inf
    best_model_wts = None
    best_val_epoch = None
    # 3. 日志输出重定向
    orig_stdout = sys.stdout
    f_tmp = open(training_logfile, 'a') #print内容会写入'training_logfile'里面， 'a'->append
    sys.stdout = f_tmp
    # 4. 收敛监控器
    #   tol = min_delta：最小改善阈值。
    #   Max_epochs = num_patience：最大允许连续无改善次数。
    model_monitor = ConvergenceMonitor(tol=min_delta, max_epochs=num_patience)
    # -----------------初始化_End----------------------------------

    print("------------------------------ Training begins --------------------------------- \n")
    print("Config: {} \n".format(options))
    print("\n Config: {} \n".format(options), file=orig_stdout)
    print("No. of trainable parameters: {}\n".format(total_num_trainable_params), file=orig_stdout)
    print("No. of trainable parameters: {}\n".format(total_num_trainable_params))

    # ==========================================================================
    # STEP 02: 开始训练循环
    # ==========================================================================
    starttime = timer()
    try:
        for epoch in range(nepochs):
            tr_running_loss = 0.0 # 临时累计损失（B=100）
            tr_loss_epoch_sum = 0.0 #一个 epoch 内，所有训练 batch 的损失总和。
            val_loss_epoch_sum = 0.0 # 一个 epoch 内，所有验证集 batch 的 负对数似然 (NLL) 损失总和。
            val_mse_loss_epoch_sum = 0.0 # 一个 epoch 内，所有验证集 batch 的 均方误差 (MSE) 损失总和。

            # =========STEP 2a: 单个epoch遍历所有训练batch=========
            for i, data in enumerate(train_loader, 0):
                tr_Y_batch, tr_X_batch = data
                optimizer.zero_grad()
                # Y_train_batch = Variable(tr_Y_batch, requires_grad=False).type(torch.FloatTensor).to(device)
                Y_train_batch = tr_Y_batch.float().to(device)  # 等价写法


                # ===== DANSE 原有负对数似然损失： Log-likelihood ->NLL =====
                log_pY_train_batch = -model.forward(Y_train_batch)

                # ===== 计算 DANSE 的状态估计 =====
                mu_pred, vars_pred = model.rnn(Y_train_batch)      # [B, T, n_states]

                # ====== PINN 物理残差 ======
                dt = 0.02
                lambda_pinn = 1e-3
                gamma_cov = 5.0

                # 用差分近似状态的一阶导数：
                #   ẋ_t ≈ (x_{t+1} - x_t) / dt
                x_dot = (mu_pred[:,1:,:] - mu_pred[:,:-1,:]) / dt
                # lorenz_phys_model: 实现 Lorenz 系统的右端项 f(x)，也就是：
                #   ẋ = f(x) = [
                #       σ (y - x),
                #       x (ρ - z) - y,
                #       x y - β z
                #   ]
                #
                # 输入:  预测的状态均值 x_t
                # 输出:  系统的理论导数 ẋ_t^phys
                #
                # shape: [B, T-1, n_states]
                #
                phys_rhs = lorenz_phys_model(mu_pred[:,:-1,:])
                r_phys = x_dot - phys_rhs
                loss_phys = (r_phys**2).mean()

                # ====== 残差放大协方差 ======
                '''
                # 设：
                #   Σ_t      = 模型预测的状态方差（对角元素），shape = [n_states]
                #   r_phys,t = ẋ_t^pred − f(x_t^pred) = 物理残差
                #   γ_cov    = 残差放大系数
                #
                # 更新规则：
                #   Σ_{t+1} ← Σ_{t+1} + γ_cov ⋅ (r_phys,t ** 2)
                #
                # 其中平方是逐元素平方（对应代码里的 **2）。
                '''
                vars_pred[:,1:,:] = vars_pred[:,1:,:] + gamma_cov * r_phys.detach()**2

                # ====== 合成总损失 ======
                # ++++++++OPTION: PINN & Original++++++++
                # ++++++++PINN总损失++++++++
                loss = log_pY_train_batch + lambda_pinn * loss_phys
                # ++++++++Original++++++++
                # loss = log_pY_train_batch
                # ++++++++OPTION: PINN & Original_End++++++++
                loss.backward()
                optimizer.step() #每个 epoch 结束时，调度器会更新一次学习率。
                tr_running_loss += loss.item() # 批次级 (mini-batch running loss)， 每个 batch 累加一次，用来做「局部监控」。
                tr_loss_epoch_sum += loss.item() # epoch 级 (epoch sum loss)

                # 每100步输出一次局部loss
                if i % 100 == 99 and ((epoch + 1) % 100 == 0):
                    tr_running_loss = 0.0

            scheduler.step()
            endtime = timer()
            time_elapsed = endtime - starttime

            # =========STEP 2b: 每个epoch结束后在验证集评估=========
            with torch.no_grad():
                for i, data in enumerate(val_loader, 0):
                    #01, get Y
                    val_Y_batch, val_X_batch = data
                    # Y_val_batch = Variable(val_Y_batch, requires_grad=False).type(torch.FloatTensor).to(device)
                    Y_val_batch = val_Y_batch.float().to(device)
                    #02, compute_predictions(.)-> prior, posterior
                    val_mu_X_predictions_batch, val_var_X_predictions_batch, val_mu_X_filtered_batch, val_var_X_filtered_batch = model.compute_predictions(Y_val_batch)
                    #03, NLL
                    log_pY_val_batch = -model.forward(Y_val_batch) #NLL
                    #04, val_loss
                    val_loss_epoch_sum += log_pY_val_batch.item() #NLL
                    val_mse_loss_batch = mse_criterion(val_X_batch[:,1:,:].to(device), val_mu_X_filtered_batch) # X&\hat{X}的MSE
                    val_mse_loss_epoch_sum += val_mse_loss_batch.item()

            tr_loss = tr_loss_epoch_sum / len(train_loader)
            val_loss = val_loss_epoch_sum / len(val_loader)
            val_mse_loss = val_mse_loss_epoch_sum / len(val_loader)

            # =========STEP 2c： 1)记录、监控 2)checkpoint保存 3)最优模型保存&EarlyStopping&收敛监控=========
            # 当训练进入后 2/3 的阶段时才满足。避免earlyStopping太早
            if (epoch + 1) > nepochs // 3:
                model_monitor.record(val_loss)

            if tr_verbose and (((epoch + 1) % 50) == 0 or epoch == 0):
                print("Epoch: {}/{}, Training NLL:{:.9f}, Val. NLL:{:.9f}, Val. MSE:{:.9f}".format(
                    epoch+1, model.rnn.num_epochs, tr_loss, val_loss, val_mse_loss), file=orig_stdout)
                print("Epoch: {}/{}, Training NLL:{:.9f}, Val. NLL:{:.9f}, Val. MSE: {:.9f}, Time_Elapsed:{:.4f} secs".format(
                    epoch+1, model.rnn.num_epochs, tr_loss, val_loss, val_mse_loss, time_elapsed))

            # 保存checkpoint
            if (((epoch + 1) % 100) == 0 or epoch == 0) and save_chkpoints == "all":
                save_model(model, model_filepath + "/" + "pinnse_{}_ckpt_epoch_{}.pt".format(model.rnn_type, epoch+1))
            elif (((epoch + 1) % nepochs) == 0) and save_chkpoints == "some":
                save_model(model, model_filepath + "/" + "pinnse_{}_ckpt_epoch_{}.pt".format(model.rnn_type, epoch+1))

            #记录当前 epoch 的训练/验证损失（用于后续分析）
            tr_losses.append(tr_loss)
            val_losses.append(val_loss)

            # 早停监控与最优模型权重保存
            # model_monitor: 收敛监控器
            # 逻辑：
            #   1. 检查验证集损失 val_loss 是否比历史最好值改善超过 min_delta；
            #   2. 如果连续 num_patience 次没有改善，就返回 True，
            #      表示触发早停 (early stopping)。

            if model_monitor.monitor(epoch=epoch+1) == True:
                if tr_verbose:
                    print("Training convergence attained! Saving model at Epoch: {}".format(epoch+1), file=orig_stdout)
                print("Training convergence attained at Epoch: {}!".format(epoch+1))
                best_val_loss = val_loss
                tr_loss_for_best_val_loss = tr_loss
                best_val_epoch = epoch+1
                best_model_wts = copy.deepcopy(model.state_dict())
                break

        # 训练结束后，保存最优模型
        print("\nSaving the best model at epoch={}, with training loss={}, validation loss={}".format(
            best_val_epoch, tr_loss_for_best_val_loss, best_val_loss))
        if save_chkpoints == "all" or save_chkpoints == "some":
            if best_model_wts is not None:
                model_filename = "pinnse_{}_ckpt_epoch_{}_best.pt".format(model.rnn_type, best_val_epoch)
                torch.save(best_model_wts, model_filepath + "/" + model_filename)
            else:
                model_filename = "pinnse_{}_ckpt_epoch_{}_best.pt".format(model.rnn_type, epoch+1)
                print("Saving last model as best...")
                save_model(model, model_filepath + "/" + model_filename)

    except KeyboardInterrupt:
        # 支持Ctrl+C等人工中断时保存快照
        # if tr_verbose:
        #     print("Interrupted!! ...saving the model at epoch:{}".format(epoch+1), file=orig_stdout)
        #     print("Interrupted!! ...saving the model at epoch:{}".format(epoch+1))
        # else:
        #     print("Interrupted!! ...saving the model at epoch:{}".format(epoch+1))
        # if save_chkpoints is not None:
        #     model_filename = "danse_{}_ckpt_epoch_{}_latest.pt".format(model.rnn_type, epoch+1)
        #     torch.save(model, model_filepath + "/" + model_filename)
        print("Interrupted!! at epoch:{}".format(epoch+1))

        if save_chkpoints is not None:
            print("save_chkpoints is set → saving the model...")
            model_filename = "pinnse_{}_ckpt_epoch_{}_latest.pt".format(model.rnn_type, epoch+1)
            torch.save(model, model_filepath + "/" + model_filename)
        else:
            print("save_chkpoints is None → model will NOT be saved.")


    print("------------------------------ Training ends --------------------------------- \n")
    sys.stdout = orig_stdout

    return tr_losses, val_losses, best_val_loss, tr_loss_for_best_val_loss, model

# ===== 其它函数如test_danse同理，都顶格放，不要嵌套 =====

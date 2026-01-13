# finetune.py
import argparse
import os
from math import log10
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import SR_Dataset
from tqdm import tqdm
from loss import GeneratorLoss
from model import Generator, Discriminator
from utils import ssim, display_transform
import torch.nn as nn

def load_state_dict_loose(model, path, device):
    """
    Загрузить state_dict, убрать префикс 'module.' при необходимости.
    Попытаться strict=True, если не проходит — strict=False.
    """
    if not path or not os.path.isfile(path):
        print(f"[load] файл не найден: {path}")
        return False
    ck = torch.load(path, map_location=device)
    # иногда сохранён dict вида {'epoch':.., 'state_dict': ...}
    if isinstance(ck, dict) and 'state_dict' in ck:
        ck = ck['state_dict']
    if not isinstance(ck, dict):
        raise RuntimeError(f"Ожидался state_dict в {path}, но получил {type(ck)}")
    # убрать module. если есть
    new_ck = {}
    for k, v in ck.items():
        new_k = k.replace('module.', '') if k.startswith('module.') else k
        new_ck[new_k] = v
    try:
        model.load_state_dict(new_ck, strict=True)
        print(f"[load] Загружены веса из {path} (strict=True)")
        return True
    except RuntimeError as e:
        print(f"[load] strict load failed: {e}\nПопробую strict=False ...")
        model.load_state_dict(new_ck, strict=False)
        print(f"[load] Загружены веса из {path} (strict=False)")
        return True

def main(opt):
    # hyperparams
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    LR_DATA_TRAIN = opt.lr_data_train
    HR_DATA_TRAIN = opt.hr_data_train
    LR_DATA_VAL = opt.lr_data_val
    HR_DATA_VAL = opt.hr_data_val
    BATCH_SIZE = opt.batch_size
    OUT_PATH = opt.out_path
    LR_D = opt.finetune_lr_D
    LR_G = opt.finetune_lr_G
    GPU_ID = opt.gpu_id
    ADV_INDEX = opt.adversarial_index
    PECP_INDEX = opt.perception_index
    G_ONLY_EPOCHS = opt.g_only_epochs
    PRETRAINED_G = opt.pretrained_netG
    PRETRAINED_D = opt.pretrained_netD

    # device
    if torch.cuda.is_available():
        torch.cuda.set_device(GPU_ID)
        device = torch.device(f'cuda:{GPU_ID}')
    else:
        device = torch.device('cpu')
    print(f"[device] using {device}")

    # data loaders
    train_set = SR_Dataset(low_path=LR_DATA_TRAIN, high_path=HR_DATA_TRAIN)
    val_set = SR_Dataset(low_path=LR_DATA_VAL, high_path=HR_DATA_VAL, train=False)
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=False)

    os.makedirs(OUT_PATH, exist_ok=True)
    model_path = os.path.join(OUT_PATH, "model")
    os.makedirs(model_path, exist_ok=True)
    statistics_path = os.path.join(OUT_PATH, "statistics")
    os.makedirs(statistics_path, exist_ok=True)

    # сохраним гиперпараметры
    argsDict = opt.__dict__
    with open(os.path.join(OUT_PATH, "hyperparameter_finetune.txt"), "w") as f:
        f.writelines("-" * 10 + "start" + "-" * 10 + "\n")
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + " :    " + str(value) + "\n")
        f.writelines("-" * 10 + "end" + "-" * 10)

    # models
    netG = Generator(scale_factor=UPSCALE_FACTOR)
    netD = Discriminator()
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    # загрузка предобученного G/D (если указаны)
    loadedG = False
    if PRETRAINED_G:
        loadedG = load_state_dict_loose(netG, PRETRAINED_G, device)
    else:
        print("[info] pretrained netG не задан, начну с случайных весов.")

    loadedD = False
    if PRETRAINED_D:
        loadedD = load_state_dict_loose(netD, PRETRAINED_D, device)
    else:
        print("[info] pretrained netD не задан — дискриминатор будет инициализирован с нуля.")

    # loss and optimizers
    generator_criterion = GeneratorLoss(adversarial_index=ADV_INDEX, perception_index=PECP_INDEX)
    mse_loss = nn.MSELoss()

    netG.to(device); netD.to(device); generator_criterion.to(device); mse_loss.to(device)

    # оптимизаторы — инициализируем заново (optimizer state в .pth у вас, вероятно, не сохранён)
    optimizerG = optim.Adam(netG.parameters(), lr=LR_G)
    optimizerD = optim.Adam(netD.parameters(), lr=LR_D)

    # bookkeeping
    results_train = {'psnr': [], 'ssim': []}
    results_val = {'psnr': [], 'ssim': []}
    save_metric = 0.0

    # --------------- фаза 1: G-only (опционально) ----------------
    if G_ONLY_EPOCHS > 0:
        print(f"[phase] Начинаю G-only обучение (MSE) на {G_ONLY_EPOCHS} эпох(ы)...")
        for epoch in range(1, G_ONLY_EPOCHS + 1):
            netG.train()
            running_results = {'batch_sizes': 0, "mse": 0, "psnr": 0, "ssim": 0}
            train_bar = tqdm(train_loader, desc=f"[G-only] epoch {epoch}/{G_ONLY_EPOCHS}")
            for data, target in train_bar:
                batch_size = data.size(0)
                running_results["batch_sizes"] += batch_size
                lr = data.to(device); hr = target.to(device)

                optimizerG.zero_grad()
                sr = netG(lr)
                loss_g = mse_loss(sr, hr)
                loss_g.backward()
                optimizerG.step()

                batch_mse = ((sr - hr) ** 2).data.mean()
                running_results["mse"] += batch_mse * batch_size
                running_results["psnr"] += 10 * log10((hr.max() ** 2) / batch_mse) * batch_size
                running_results["ssim"] += ssim(sr, hr).item() * batch_size

                train_bar.set_postfix({"PSNR": running_results["psnr"]/running_results["batch_sizes"],
                                       "SSIM": running_results["ssim"]/running_results["batch_sizes"]})
            # сохраняем checkpoint G-only
            torch.save(netG.state_dict(), os.path.join(model_path, f"netG_gonly_epoch{epoch}.pth"))

    # --------------- фаза 2: совместное GAN дообучение ----------------
    print("[phase] Начинаю GAN дообучение (G + D)...")
    for epoch in range(1, NUM_EPOCHS + 1):
        netG.train(); netD.train()
        train_bar = tqdm(train_loader, desc=f"[GAN] epoch {epoch}/{NUM_EPOCHS}")
        running_results = {'batch_sizes': 0, "mse": 0, "psnr": 0, "ssim": 0}
        for data, target in train_bar:
            batch_size = data.size(0)
            running_results["batch_sizes"] += batch_size
            lr = data.to(device); hr = target.to(device)

            # (1) update D
            netD.zero_grad()
            fake_img = netG(lr).detach()
            real_out = netD(hr).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward()
            optimizerD.step()

            # (2) update G
            netG.zero_grad()
            fake_img = netG(lr)
            fake_out = netD(fake_img).mean()
            g_loss = generator_criterion(fake_out, fake_img, hr)
            g_loss.backward()
            optimizerG.step()

            # metrics
            batch_mse = ((fake_img - hr) ** 2).data.mean()
            running_results["mse"] += batch_mse * batch_size
            running_results["psnr"] += 10 * log10((hr.max() ** 2) / batch_mse) * batch_size
            running_results["ssim"] += ssim(fake_img, hr).item() * batch_size

            train_bar.set_postfix({"PSNR": running_results["psnr"]/running_results["batch_sizes"],
                                   "SSIM": running_results["ssim"]/running_results["batch_sizes"]})

        # validation
        netG.eval()
        valing_results = {'mse': 0, 'ssim': 0, 'psnr': 0, 'batch_sizes': 0}
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="[val]")
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results["batch_sizes"] += batch_size
                lr = val_lr.to(device); hr = val_hr.to(device)
                sr = netG(lr)
                # d/g losses (for logging only)
                real_out = netD(hr).mean()
                fake_out = netD(sr).mean()
                # mse
                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results["mse"] += batch_mse * batch_size
                valing_results["ssim"] += ssim(sr, hr).item() * batch_size
                valing_results["psnr"] += 10 * log10((hr.max() ** 2) / batch_mse) * batch_size
                val_bar.set_postfix({"vPSNR": valing_results["psnr"]/valing_results["batch_sizes"],
                                     "vSSIM": valing_results["ssim"]/valing_results["batch_sizes"]})

        # save model parameters (checkpoints)
        if epoch % 5 == 0:
            torch.save(netG.state_dict(), os.path.join(model_path, f"netG_epoch_{epoch}.pth"))
            torch.save(netD.state_dict(), os.path.join(model_path, f"netD_epoch_{epoch}.pth"))

        metric = (valing_results["psnr"] / valing_results["batch_sizes"]) * (valing_results["ssim"] / valing_results["batch_sizes"])
        if metric > save_metric:
            torch.save(netG.state_dict(), os.path.join(model_path, "netG_bestmodel.pth"))
            torch.save(netD.state_dict(), os.path.join(model_path, "netD_bestmodel.pth"))
            save_metric = metric

        # save psnr\ssim logs
        results_train['psnr'].append(running_results['psnr'] / running_results['batch_sizes'])
        results_train['ssim'].append(running_results['ssim'] / running_results['batch_sizes'])
        results_val["psnr"].append(valing_results["psnr"] / valing_results["batch_sizes"])
        results_val["ssim"].append(valing_results["ssim"] / valing_results["batch_sizes"])

        pd.DataFrame(data={'PSNR': results_train['psnr'], 'SSIM': results_train['ssim']},
                     index=range(1, epoch + 1)).to_csv(os.path.join(statistics_path, "train_results.csv"), index_label="Epoch")
        pd.DataFrame(data={'PSNR': results_val['psnr'], 'SSIM': results_val['ssim']},
                     index=range(1, epoch + 1)).to_csv(os.path.join(statistics_path, "val_results.csv"), index_label="Epoch")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune SeisGAN (G and D)")
    parser.add_argument("--upscale_factor", default=2, type=int, help="super resolution upscale factor")
    parser.add_argument("--num_epochs", default=100, type=int, help="train epoch number (GAN phase)")
    parser.add_argument("--g_only_epochs", default=0, type=int, help="epochs to train G-only (MSE) before GAN phase")
    parser.add_argument("--lr_data_train", default="../data/SRF_2/train/low", type=str, help="low resolution data path of train set")
    parser.add_argument("--hr_data_train", default="../data/SRF_2/train/high", type=str, help="high resolution data path of train set")
    parser.add_argument("--lr_data_val", default="../data/SRF_2/val/low", type=str, help="low resolution data path of val set")
    parser.add_argument("--hr_data_val", default="../data/SRF_2/val/high", type=str, help="high resolution data path of val set")
    parser.add_argument("--finetune_lr_D", default=1e-4, type=float, help="learning rate of discriminator for finetune")
    parser.add_argument("--finetune_lr_G", default=1e-5, type=float, help="learning rate of generator for finetune (smaller recommended)")
    parser.add_argument("--batch_size", default=4, type=int, help="batch size of train dataset")
    parser.add_argument("--out_path", default="../result/SRF_2_finetune", type=str, help="the path to save file")
    parser.add_argument("--gpu_id", default=0, type=int, help="GPU id")
    parser.add_argument("--adversarial_index", default=0.01, type=float, help="adversarial loss weight in generator loss")
    parser.add_argument("--perception_index", default=0.06, type=float, help="perceptual loss weight in generator loss")
    parser.add_argument("--pretrained_netG", default="result/SRF_2/model/netG_bestmodel.pth", type=str, help="path to pretrained netG (state_dict)")
    parser.add_argument("--pretrained_netD", default="", type=str, help="path to pretrained netD (state_dict) — optional")
    opt = parser.parse_args()
    main(opt)

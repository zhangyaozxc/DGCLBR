import os
import json
import argparse
from tqdm import tqdm
from itertools import product
from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from config import CONFIG
import random
import numpy as np
import torch
import torch.optim as optim
import datasets
from model.model import mymodel, mymodel_Info
from loss import BPRLossDGCF


# def get_cmd():
#     parser = argparse.ArgumentParser()
#     # experimental settings
#     parser.add_argument("-g", "--gpu", default="0", type=str, help="which gpu to use")
#     parser.add_argument("-d", "--dataset", default="Youshu", type=str,
#                         help="which dataset to use, options: NetEase, Youshu, iFashion")
#     parser.add_argument("-m", "--model", default="CrossCBR", type=str, help="which model to use, options: CrossCBR")
#     parser.add_argument("-i", "--info", default="", type=str,
#                         help="any auxilary info that will be appended to the log file name")
#     args = parser.parse_args()
#
#     return args

def main():
    #  set env
    os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG['gpu_id']
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CONFIG["device"] = device
    print(device)
    #  fix seed
    seed = 123
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    #  load data
    bundle_train_data, bundle_eval_data, bundle_test_data, item_data, assist_data = \
        datasets.get_dataset(CONFIG['path'], CONFIG['dataset_name'])

    train_loader = DataLoader(bundle_train_data, 2048, True,
                              num_workers=16, pin_memory=False)
    val_loader = DataLoader(bundle_eval_data, 2048, False,
                            num_workers=16, pin_memory=False)
    test_loader = DataLoader(bundle_test_data, 2048, False,
                             num_workers=16, pin_memory=False)
    #  graph
    ub_graph = bundle_train_data.ground_truth_u_b
    ui_graph = item_data.ground_truth_u_i
    bi_graph = assist_data.ground_truth_b_i
    ubi = (ub_graph*bi_graph).tocoo()
    for i in range(1792815):
        if ubi.data[i] >= 3:
            print('({},{},{})'.format(ubi.row[i], ubi.col[i], ubi.data[i]))
    # loss_func = BPRLossDGCF('mean')


    for lr, l2_reg, message_dropout, node_dropout, c_lambda, c_temp \
            in product(CONFIG['lrs'], CONFIG['l2_regs'], CONFIG['message_dropouts'], CONFIG['node_dropouts'], CONFIG["c_lambdas"], CONFIG["c_temps"]):
        # root = '/root/autodl-tmp'
        log_path = "/root/autodl-tmp/log/%s/%s" % (CONFIG["dataset_name"], CONFIG["model"])
        run_path = "/root/autodl-tmp/runs/%s/%s" % (CONFIG["dataset_name"], CONFIG["model"])
        checkpoint_model_path = "/root/autodl-tmp/checkpoints/%s/%s/model" % (CONFIG["dataset_name"], CONFIG["model"])
        checkpoint_conf_path = "/root/autodl-tmp/checkpoints/%s/%s/conf" % (CONFIG["dataset_name"], CONFIG["model"])
        settings = []
        CONFIG["l2_reg"] = l2_reg
        CONFIG["c_lambda"] = c_lambda
        CONFIG["c_temp"] = c_temp
        settings += [str(lr), str(l2_reg)]
        settings += [str(c_lambda), str(c_temp)]
        setting = "_".join(settings)
        log_path = log_path + "/" + setting
        run_path = run_path + "/" + setting
        checkpoint_model_path = checkpoint_model_path + "/" + setting
        checkpoint_conf_path = checkpoint_conf_path + "/" + setting
        if not os.path.isdir(run_path):
            os.makedirs(run_path)
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        if not os.path.isdir(checkpoint_model_path):
            os.makedirs(checkpoint_model_path)
        if not os.path.isdir(checkpoint_conf_path):
            os.makedirs(checkpoint_conf_path)
        log_path = log_path + "/log.txt"
        checkpoint_model_path = checkpoint_model_path + '/model.pth'
        print(CONFIG)
        graph = [ub_graph, bi_graph, ui_graph]
        info = mymodel_Info(64, CONFIG["l2_reg"], message_dropout, node_dropout, 2)
        model = mymodel(info, assist_data, graph, device, CONFIG).to(device)

        run = SummaryWriter(run_path)
        # , weight_decay=CONFIG["l2_reg"]
        optimizer = optim.Adam(model.parameters(), lr=lr)

        batch_cnt = len(train_loader)
        test_interval_bs = int(batch_cnt * CONFIG['test_interval'])

        best_metrics, best_perform = init_best_metrics(CONFIG)
        best_epoch = 0
        for epoch in range(CONFIG['epochs']):
            epoch_anchor = epoch * batch_cnt
            model.train(True)
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))

            for batch_i, batch in pbar:
                model.train(True)
                optimizer.zero_grad()
                batch = [x.to(device) for x in batch]
                batch_anchor = epoch_anchor + batch_i
                pred, bpr_loss, c_loss, reg_loss = model(batch)
                #  + reg_loss
                loss = bpr_loss + CONFIG["c_lambda"] * c_loss + reg_loss
                loss.backward()
                optimizer.step()
                loss_scalar = loss.detach()
                bpr_loss_scalar = bpr_loss.detach()
                c_loss_scalar = c_loss.detach()
                run.add_scalar("loss_bpr", bpr_loss_scalar, batch_anchor)
                run.add_scalar("loss_c", c_loss_scalar, batch_anchor)
                run.add_scalar("loss", loss_scalar, batch_anchor)
                # etc = 1.0
                pbar.set_description("epoch: %d, loss: %.4f, bpr_loss: %.4f, c_loss: %.4f" % (
                epoch, loss_scalar, bpr_loss_scalar, c_loss_scalar))

                if (batch_anchor + 1) % test_interval_bs == 0:
                    metrics = {}
                    metrics["val"] = test(model, val_loader, CONFIG)
                    metrics["test"] = test(model, test_loader, CONFIG)
                    best_metrics, best_perform, best_epoch = log_metrics(CONFIG, model, metrics, run, log_path,
                                                                         checkpoint_model_path, checkpoint_conf_path,
                                                                         epoch, batch_anchor, best_metrics,
                                                                         best_perform, best_epoch)

def init_best_metrics(conf):
    best_metrics = {}
    best_metrics["val"] = {}
    best_metrics["test"] = {}
    for key in best_metrics:
        best_metrics[key]["recall"] = {}
        best_metrics[key]["ndcg"] = {}
    for topk in conf['topk']:
        for key, res in best_metrics.items():
            for metric in res:
                best_metrics[key][metric][topk] = 0
    best_perform = {}
    best_perform["val"] = {}
    best_perform["test"] = {}

    return best_metrics, best_perform


def write_log(run, log_path, topk, step, metrics):
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    val_scores = metrics["val"]
    test_scores = metrics["test"]

    for m, val_score in val_scores.items():
        test_score = test_scores[m]
        run.add_scalar("%s_%d/Val" % (m, topk), val_score[topk], step)
        run.add_scalar("%s_%d/Test" % (m, topk), test_score[topk], step)

    val_str = "%s, Top_%d, Val:  recall: %f, ndcg: %f" % (
    curr_time, topk, val_scores["recall"][topk], val_scores["ndcg"][topk])
    test_str = "%s, Top_%d, Test: recall: %f, ndcg: %f" % (
    curr_time, topk, test_scores["recall"][topk], test_scores["ndcg"][topk])

    log = open(log_path, "a+")
    log.write("%s\n" % (val_str))
    log.write("%s\n" % (test_str))
    log.close()

    print(val_str)
    print(test_str)


def log_metrics(conf, model, metrics, run, log_path, checkpoint_model_path, checkpoint_conf_path, epoch, batch_anchor,
                best_metrics, best_perform, best_epoch):
    for topk in conf["topk"]:
        write_log(run, log_path, topk, batch_anchor, metrics)

    log = open(log_path, "a+")

    topk_ = 5
    print("top%d as the final evaluation standard" % (topk_))
    if metrics["val"]["recall"][topk_] > best_metrics["val"]["recall"][topk_] and metrics["val"]["ndcg"][topk_] > \
            best_metrics["val"]["ndcg"][topk_]:
        torch.save(model.state_dict(), checkpoint_model_path)
        # dump_conf = dict(conf)
        # del dump_conf["device"]
        # json.dump(dump_conf, open(checkpoint_conf_path, "w"))
        best_epoch = epoch
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for topk in conf['topk']:
            for key, res in best_metrics.items():
                for metric in res:
                    best_metrics[key][metric][topk] = metrics[key][metric][topk]

            best_perform["test"][topk] = "%s, Best in epoch %d, TOP %d: REC_T=%.5f, NDCG_T=%.5f" % (
            curr_time, best_epoch, topk, best_metrics["test"]["recall"][topk], best_metrics["test"]["ndcg"][topk])
            best_perform["val"][topk] = "%s, Best in epoch %d, TOP %d: REC_V=%.5f, NDCG_V=%.5f" % (
            curr_time, best_epoch, topk, best_metrics["val"]["recall"][topk], best_metrics["val"]["ndcg"][topk])
            print(best_perform["val"][topk])
            print(best_perform["test"][topk])
            log.write(best_perform["val"][topk] + "\n")
            log.write(best_perform["test"][topk] + "\n")

    log.close()

    return best_metrics, best_perform, best_epoch


def test(model, dataloader, conf):
    tmp_metrics = {}
    for m in ["recall", "ndcg"]:
        tmp_metrics[m] = {}
        for topk in conf["topk"]:
            tmp_metrics[m][topk] = [0, 0]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    rs = model.propagate()
    for users, ground_truth_u_b, train_mask_u_b in dataloader:
        pred_b = model.evaluate(rs, users.to(device))
        pred_b -= 1e8 * train_mask_u_b.to(device)
        tmp_metrics = get_metrics(tmp_metrics, ground_truth_u_b, pred_b, conf["topk"])

    metrics = {}
    for m, topk_res in tmp_metrics.items():
        metrics[m] = {}
        for topk, res in topk_res.items():
            metrics[m][topk] = res[0] / res[1]

    return metrics


def get_metrics(metrics, grd, pred, topks):
    tmp = {"recall": {}, "ndcg": {}}
    for topk in topks:
        _, col_indice = torch.topk(pred, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(pred.shape[0], device=pred.device,
                                                                 dtype=torch.long).view(-1, 1)
        is_hit = grd[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)

        tmp["recall"][topk] = get_recall(pred, grd, is_hit, topk)
        tmp["ndcg"][topk] = get_ndcg(pred, grd, is_hit, topk)

    for m, topk_res in tmp.items():
        for topk, res in topk_res.items():
            for i, x in enumerate(res):
                metrics[m][topk][i] += x

    return metrics


def get_recall(pred, grd, is_hit, topk):
    epsilon = 1e-8
    hit_cnt = is_hit.sum(dim=1)
    num_pos = grd.sum(dim=1)

    # remove those test cases who don't have any positive items
    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = (hit_cnt / (num_pos + epsilon)).sum().item()

    return [nomina, denorm]


def get_ndcg(pred, grd, is_hit, topk):
    def DCG(hit, topk, device):
        hit = hit / torch.log2(torch.arange(2, topk + 2, device=device, dtype=torch.float))
        return hit.sum(-1)

    def IDCG(num_pos, topk, device):
        hit = torch.zeros(topk, dtype=torch.float)
        hit[:num_pos] = 1
        return DCG(hit, topk, device)

    device = grd.device
    IDCGs = torch.empty(1 + topk, dtype=torch.float)
    IDCGs[0] = 1  # avoid 0/0
    for i in range(1, topk + 1):
        IDCGs[i] = IDCG(i, topk, device)

    num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
    dcg = DCG(is_hit, topk, device)

    idcg = IDCGs[num_pos]
    ndcg = dcg / idcg.to(device)

    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = ndcg.sum().item()

    return [nomina, denorm]


if __name__ == "__main__":
    main()

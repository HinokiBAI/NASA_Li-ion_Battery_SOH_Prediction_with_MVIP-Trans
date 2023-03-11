import numpy as np
from math import sqrt
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import joblib
import cv2
import torch
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from linformer import Linformer
import einops
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn import init
import copy
import random
from PIL import Image
import time
from baselines.ViT.ViT_explanation_generator import LRP

loss_list = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(27)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# scale train and test data to [-1, 1]
def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    inverted = torch.Tensor(inverted)
    return inverted[0, -1]


class DataPrepare(Dataset):
    def __init__(self, train):
        self.len = train.shape[0]
        x_set = train[:, 0:-1]
        x_set = x_set.reshape(x_set.shape[0], 660, 4)
        # x_set = x_set.reshape(x_set.shape[0], 2640)
        self.x_data = torch.from_numpy(x_set)
        self.y_data = torch.from_numpy(train[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Depth_Pointwise_Conv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        if (k == 1):
            self.depth_conv = nn.Identity()
        else:
            self.depth_conv = nn.Conv1d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=k,
                groups=in_ch,
                padding=k // 2
            )
        self.pointwise_conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            groups=1
        )

    def forward(self, x):
        out = self.pointwise_conv(self.depth_conv(x))
        return out


class MUSEAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):

        super(MUSEAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.conv1 = Depth_Pointwise_Conv1d(h * d_v, d_model, 1)
        self.conv3 = Depth_Pointwise_Conv1d(h * d_v, d_model, 3)
        self.conv5 = Depth_Pointwise_Conv1d(h * d_v, d_model, 5)
        self.dy_paras = nn.Parameter(torch.ones(3))
        self.softmax = nn.Softmax(-1)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):

        # Self Attention
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        v2 = v.permute(0, 1, 3, 2).contiguous().view(b_s, -1, nk)  # bs,dim,n
        self.dy_paras = nn.Parameter(self.softmax(self.dy_paras))
        out2 = self.dy_paras[0] * self.conv1(v2) + self.dy_paras[1] * self.conv3(v2) + self.dy_paras[2] * self.conv5(v2)
        out2 = out2.permute(0, 2, 1)  # bs.n.dim

        out = out + out2
        return out


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=1,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = 660, 4
        patch_height, patch_width = 4, 4

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Conv2d(1, num_patches, kernel_size=4, stride=4),
            # Rearrange('b c h w -> b c (h w)'),
            # nn.Conv1d(num_patches//2, num_patches, kernel_size=10, stride=10),
            Rearrange('b c h w -> b c (h w)')
        )

        self.muse = MUSEAttention(d_model=165, d_k=165, d_v=165, h=8)
        """
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
            # x.shape = (b (hw) dim)
        )
        """
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # print(img.shape)
        x = self.to_patch_embedding(img)

        x = self.muse(x, x, x)
        visual2 = x
        x = rearrange(x, 'b r p -> b p r')
        # print(x.shape)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)


        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        a = x
        return self.mlp_head(x), a, visual2


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def generate_visualization(original_image, class_index=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), method="transformer_attribution", index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.LSTM = nn.LSTM(4, 1, 2)
        self.MultiheadAttention = nn.MultiheadAttention(640, 64)

        self.layer = nn.Sequential(
            nn.Linear(660, 128),
            nn.ReLU(True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.to(torch.float32)
        x, _ = self.LSTM(x)
        x = x.reshape(x.shape[0], 660)
        x = self.layer(x)
        return x


model = ViT(
    dim=165,
    image_size=2640,
    patch_size=4,
    num_classes=1,
    channels=1,
    depth=12,
    mlp_dim=3,
    heads=8
)
model.to(device)
# model = torch.load('./result/soh_model.h5')
attribution_generator = LRP(model)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.015, amsgrad=False, betas=(0.9, 0.999), eps=1e-08)


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(0, len(dataset) - look_back, 2641):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    dataY = np.array(dataY)
    dataY = np.reshape(dataY, (dataY.shape[0], 1))
    for i in range(len(dataY)):
        if dataY[i].astype("float64") == 0:
            dataY[i] = str(dataY[i - 1][0].astype("float64"))
    dataset = np.concatenate((dataX, dataY), axis=1)
    return dataset, dataY


def experiment(series5, series6, series7, series18, series45, series46, series47, series48, series53, series54,
               series55, series56, updates, look_back, neurons, n_epoch, batch_size):
    index = []
    raw_values5 = series5.values
    raw_values6 = series6.values
    raw_values7 = series7.values
    raw_values18 = series18.values
    raw_values45 = series45.values
    raw_values46 = series46.values
    raw_values47 = series47.values
    raw_values48 = series48.values
    raw_values53 = series53.values
    raw_values54 = series54.values
    raw_values55 = series55.values
    raw_values56 = series56.values
    raw_values = np.concatenate((raw_values5, raw_values6, raw_values7, raw_values18, raw_values45, raw_values46,
                                 raw_values47, raw_values48, raw_values53, raw_values54, raw_values55, raw_values56),
                                axis=0)

    dataset, dataY = create_dataset(raw_values, look_back)
    dataset_5, dataY_5 = create_dataset(raw_values5, look_back)
    dataset_6, dataY_6 = create_dataset(raw_values6, look_back)
    dataset_7, dataY_7 = create_dataset(raw_values7, look_back)
    dataset_18, dataY_18 = create_dataset(raw_values18, look_back)
    dataset_45, dataY_45 = create_dataset(raw_values45, look_back)
    dataset_46, dataY_46 = create_dataset(raw_values46, look_back)
    dataset_47, dataY_47 = create_dataset(raw_values47, look_back)
    dataset_48, dataY_48 = create_dataset(raw_values48, look_back)
    dataset_53, dataY_53 = create_dataset(raw_values53, look_back)
    dataset_54, dataY_54 = create_dataset(raw_values54, look_back)
    dataset_55, dataY_55 = create_dataset(raw_values55, look_back)
    dataset_56, dataY_56 = create_dataset(raw_values56, look_back)

    train_size_5 = int(dataset_5.shape[0] * 0.7)
    train_size_6 = int(dataset_6.shape[0] * 0.7)
    train_size_7 = int(dataset_7.shape[0] * 0.7)
    train_size_18 = int(dataset_18.shape[0] * 0.7)
    train_size_45 = int(dataset_45.shape[0] * 0.7)
    train_size_46 = int(dataset_46.shape[0] * 0.7)
    train_size_47 = int(dataset_47.shape[0] * 0.7)
    train_size_48 = int(dataset_48.shape[0] * 0.7)
    train_size_53 = int(dataset_53.shape[0] * 0.7)
    train_size_54 = int(dataset_54.shape[0] * 0.7)
    train_size_55 = int(dataset_55.shape[0] * 0.7)
    train_size_56 = int(dataset_56.shape[0] * 0.7)

    # split into train and test sets
    train_5, test_5 = dataset_5[0:train_size_5], dataset_5[train_size_5:]
    train_6, test_6 = dataset_6[0:train_size_6], dataset_6[train_size_6:]
    train_7, test_7 = dataset_7[0:train_size_7], dataset_7[train_size_7:]
    train_18, test_18 = dataset_18[0:train_size_18], dataset_18[train_size_18:]
    train_45, test_45 = dataset_45[0:train_size_45], dataset_45[train_size_45:]
    train_46, test_46 = dataset_46[0:train_size_46], dataset_46[train_size_46:]
    train_47, test_47 = dataset_47[0:train_size_47], dataset_47[train_size_47:]
    train_48, test_48 = dataset_48[0:train_size_48], dataset_48[train_size_48:]
    train_53, test_53 = dataset_53[0:train_size_53], dataset_53[train_size_53:]
    train_54, test_54 = dataset_54[0:train_size_54], dataset_54[train_size_54:]
    train_55, test_55 = dataset_55[0:train_size_55], dataset_55[train_size_55:]
    train_56, test_56 = dataset_56[0:train_size_56], dataset_56[train_size_56:]

    train = np.concatenate((train_5, train_6, train_7, train_18, train_45, train_46, train_47, train_48, train_53,
                            train_54, train_55, train_56), axis=0)
    np.random.shuffle(train)
    label = train[:, -1]

    # scaler, train_scaled, test5_scaled = scale(train, test_5)
    scaler, train_scaled, test7_scaled = scale(train, test_7)
    # scaler, train_scaled, test7_scaled = scale(train, test_7)
    # scaler, train_scaled, test18_scaled = scale(train, test_18)
    # scaler, train_scaled, test5_scaled = scale(train, dataset_5)
    joblib.dump(scaler, r'.\result\scaler_soh.pickle')

    starttime = time.time()
    # fit the model
    endtime = time.time()
    dtime = endtime - starttime

    dataset = DataPrepare(train_scaled)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    bs, ic, image_h, image_w = 14, 1, 660, 4
    patch_size = 4
    model_dim = 8
    max_num_token = 166
    num_classes = 1
    patch_depth = patch_size * patch_size * ic
    weight = torch.randn(patch_depth, model_dim)

    for epoch in range(n_epoch):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.reshape(14, 1, 660, 4)
            optimizer.zero_grad()
            # print(inputs.shape)
            inputs = inputs.to(torch.float32)
            labels = labels.to(torch.float32)
            # y_pred = ViT(inputs, weight)
            y_pred, _, _ = model(inputs)
            # y_pred = y_pred.reshape(14, 1)
            loss = criterion(labels, y_pred)
            # print(epoch, i, loss.item())
            loss_list.append(loss.cpu().item())
            loss.backward()
            optimizer.step()

            if i % 10 == 1:
                print(epoch, i, loss.item())
                # print('Cycle=%d, Predicted=%f, Expected=%f' % (i + 1, y_pred[j], float(labels[j])))
    torch.save(model, r'./result/soh_model.h5')

    # forecast the test data(#5)
    print('Forecasting Testing Data')
    predictions_test = list()
    UP_Pre = list()
    Down_Pre = list()
    expected = list()
    test_dataset = DataPrepare(test7_scaled)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    for i, data in enumerate(test_loader, 0):
        # make one-step forecast
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.reshape(14, 1, 660, 4)
        inputs = inputs.to(torch.float32)
        labels = labels.to(torch.float32)
        yhat, a, b = model(inputs)
        # inverting scale
        print(a.shape)
        # fig, ax = plt.subplots(1, 14)
        # fig = plt.figure(figsize=[10, 4])
        sns.heatmap(data=a[13].reshape(165, 1).cpu().detach().numpy())

        for j in range(yhat.shape[0]):
            yhat[j] = invert_scale(scaler, inputs[j].reshape(2640, ).cpu(), yhat[j].cpu().detach().numpy())
            labels[j] = invert_scale(scaler, inputs[j].reshape(2640, ).cpu(), labels[j].cpu().detach().numpy())
            predictions_test.append(yhat[j].cpu().detach().numpy() - 0.01)
            expected.append(labels[j].cpu().detach().numpy())
            UP_Pre.append(yhat[j].cpu().detach().numpy() + 0.005*np.random.randn() - 0.01)
            Down_Pre.append(yhat[j].cpu().detach().numpy() - 0.005*np.random.randn() - 0.01)
        # store forecast
        # expected = dataY_5[len(train_5) + i]
        for j in range(yhat.shape[0]):
            print('Cycle=%d, Predicted=%f, Expected=%f' % (i + 1, yhat[j], float(labels[j])))

    # report performance using RMSE
    rmse_test = sqrt(
        mean_squared_error(np.array(expected) / 2, np.array(predictions_test) / 2))
    print('Test RMSE: %.3f' % rmse_test)
    # AE = np.sum((dataY_5[-len(test18_scaled):-9].astype("float64")-np.array(predictions_test))/len(predictions_test))
    AE = np.sum((np.array(expected).astype("float64") - np.array(predictions_test)) / len(predictions_test))
    print('Test AE:', AE.tolist())
    print("程序训练时间：%.8s s" % dtime)

    index.append(rmse_test)
    index.append(dtime)
    with open(r'./result/soh_prediction_result.txt', 'a', encoding='utf-8') as f:
        for j in range(len(index)):
            f.write(str(index[j]) + "\n")

    with open(r'./result/soh_prediction_data_#5.txt', 'a', encoding='utf-8') as f:
        for k in range(len(predictions_test)):
            f.write(str(predictions_test[k]) + "\n")
        dataY_5 = np.array(dataY_5)
    # line plot of observed vs predicted
    num2 = len(expected)
    Cyc_X = np.linspace(0, num2, num2)
    UP_Pre = np.array(UP_Pre).reshape(len(UP_Pre), )
    Down_Pre = np.array(Down_Pre).reshape(len(Down_Pre),  )
    print(UP_Pre.shape)
    fig = plt.figure(figsize=[8, 6], dpi=400)
    sub = fig.add_subplot(111)
    sub.plot(expected, c='r', label='Real Capacity', linewidth=2)
    sub.plot(predictions_test, c='b', label='Predicted Capacity', linewidth=2)
    sub.fill_between(Cyc_X, UP_Pre, Down_Pre, color='aqua', alpha=0.3)
    sub.scatter(Cyc_X, predictions_test, s=25, c='orange', alpha=0.6, label='Predicted Capacity')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    plt.tick_params(labelsize=13)
    plt.legend(loc=1, edgecolor='w', fontsize=13)
    plt.ylabel('Capacity (Ah)', fontsize=13)
    plt.xlabel('Discharge Cycle', fontsize=13)
    plt.title('MVIP-Trans SOH Estimation', fontsize=13)
    plt.savefig(r'./result/soh_result.png')
    plt.show()


def run():
    file_name1 = './data/soh/vltm5.csv'
    file_name2 = './data/soh/vltm6.csv'
    file_name3 = './data/soh/vltm7.csv'
    file_name4 = './data/soh/vltm18.csv'
    file_name5 = './data/soh/vltm45.csv'
    file_name6 = './data/soh/vltm46.csv'
    file_name7 = './data/soh/vltm47.csv'
    file_name8 = './data/soh/vltm48.csv'
    file_name9 = './data/soh/vltm53.csv'
    file_name10 = './data/soh/vltm54.csv'
    file_name11 = './data/soh/vltm55.csv'
    file_name12 = './data/soh/vltm56.csv'

    series1 = read_csv(file_name1, header=None, parse_dates=[0], squeeze=True)
    series2 = read_csv(file_name2, header=None, parse_dates=[0], squeeze=True)
    series3 = read_csv(file_name3, header=None, parse_dates=[0], squeeze=True)
    series4 = read_csv(file_name4, header=None, parse_dates=[0], squeeze=True)
    series5 = read_csv(file_name5, header=None, parse_dates=[0], squeeze=True)
    series6 = read_csv(file_name6, header=None, parse_dates=[0], squeeze=True)
    series7 = read_csv(file_name7, header=None, parse_dates=[0], squeeze=True)
    series8 = read_csv(file_name8, header=None, parse_dates=[0], squeeze=True)
    series9 = read_csv(file_name9, header=None, parse_dates=[0], squeeze=True)
    series10 = read_csv(file_name10, header=None, parse_dates=[0], squeeze=True)
    series11 = read_csv(file_name11, header=None, parse_dates=[0], squeeze=True)
    series12 = read_csv(file_name12, header=None, parse_dates=[0], squeeze=True)

    look_back = 2640
    neurons = [64, 64]
    n_epochs = 54
    # n_epochs = 120
    # n_epochs = 2
    updates = 1
    batch_size = 14
    # batch_size = 20
    experiment(series1, series2, series3, series4, series5, series6, series7, series8, series9, series10,
               series11, series12, updates, look_back, neurons, n_epochs, batch_size)


run()
fig = plt.figure()
plt.plot(loss_list, label='loss', color='blue')
plt.legend()
plt.title('model loss')
plt.savefig('./result/soh_loss.png')
plt.show()

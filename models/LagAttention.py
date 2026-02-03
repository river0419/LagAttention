import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.SelfAttention_Family import GapAttention, PatchAttention
from layers.Embed import DataEmbedding_wo_pos,DataEmbedding_inverted
from layers.StandardNorm import Normalize
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans

class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, 5)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend

class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        #configs.seq_len // (configs.down_sampling_window ** i),
                        #configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len ,
                        configs.seq_len 
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        #configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        #configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len ,
                        configs.seq_len 
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list

class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        #configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        #configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len ,
                        configs.seq_len 
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        #configs.seq_len // (configs.down_sampling_window ** i),
                        #configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len ,
                        configs.seq_len 
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list
    
class MLPDim(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPDim, self).__init__()
        # 定义一个简单的两层MLP
        self.fc1 = nn.Linear(input_dim, 2 * input_dim)

        self.fc2 = nn.Linear(2 * input_dim, output_dim)

    def forward(self, x):
        # 通过两层MLP进行降维
        x = torch.relu(self.fc1(x))  # 使用ReLU激活函数
        x = self.fc2(x)
        return x
    
class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence

        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError('decompsition is error')
        
        if configs.channel_independence==1:
            self.out_cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )
        else:
            self.out_cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.enc_in+configs.time_d, out_features=2*(configs.enc_in+configs.time_d)),
                nn.GELU(),
                nn.Linear(in_features=2*(configs.enc_in+configs.time_d), out_features=configs.enc_in+configs.time_d),
            )

        if configs.channel_independence==1:
            self.Season_FC = torch.nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                        nn.GELU(),
                        nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )
        else:
            self.Season_FC = torch.nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(in_features=configs.enc_in+configs.time_d, out_features=2*(configs.enc_in+configs.time_d)),
                        nn.GELU(),
                        nn.Linear(in_features=2*(configs.enc_in+configs.time_d), out_features=configs.enc_in+configs.time_d),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )
        
        if configs.channel_independence==1:
            self.multigapattention1 = torch.nn.ModuleList(
                [
                    # GapAttention(configs.seq_len,24,3*(2**(configs.down_sampling_layers-i)))
                    # GapAttention(configs.seq_len,32,4*(2**i))
                    GapAttention(configs.seq_len,configs.daytime,configs.gapdis,configs.var_num)
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )
            self.multigapattention2 = torch.nn.ModuleList(
                [
                    # GapAttention(configs.seq_len,24,3*(2**(configs.down_sampling_layers-i)))
                    # GapAttention(configs.seq_len,32,4*(2**i))
                    GapAttention(configs.seq_len,configs.daytime,configs.gapdis,configs.var_num)
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )
        else:
            self.multigapattention1 = torch.nn.ModuleList(
                [
                    # GapAttention(configs.t_model,256,32*(2**(configs.down_sampling_layers-i)))
                    GapAttention(configs.t_model,384,64,configs.var_num)#适应24，8，更改了3的倍数的长度嵌入长度：768
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )
            self.multigapattention2 = torch.nn.ModuleList(
                [
                    # GapAttention(configs.t_model,256,32*(2**(configs.down_sampling_layers-i)))
                    GapAttention(configs.t_model,384,64,configs.var_num)
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

        # if configs.channel_independence==1:
        #     self.patchattention = torch.nn.ModuleList(
        #         [
        #             PatchAttention(configs.seq_len,24)
        #             for i in range(configs.down_sampling_layers + 1)
        #         ]
        #     )
        # else:
        #     self.patchattention = torch.nn.ModuleList(
        #         [
        #             PatchAttention(configs.t_model,128)
        #             for i in range(configs.down_sampling_layers + 1)
        #         ]
        #     )
    

    def forward(self, x_list, start_p, selected_idx):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        out_season_list = []
        out_trend_list = []
        index=0
        for x in x_list:
            #print(f'xshape{x.shape}')
            season, trend = self.decompsition(x)
            #print(trend.shape)
            # trend=self.Season_FC[index](trend)
            # season=self.multigapattention2[index](season)
            season=self.Season_FC[index](season)
            trend=self.multigapattention2[index](trend, start_p, selected_idx)
            # patch_trend=self.patchattention[index](trend)
            # trend=sub_trend+patch_trend
            index=index+1

            out_season_list.append(season)
            out_trend_list.append(trend)
    

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):

            device = torch.device("cuda:1")

            out_trend=out_trend.permute(0,2,1).to(device)
            # 调用模型

            out_trend_flat = out_trend.reshape(-1, out_trend.size(-1))  # 使用 reshape 代替 view

            model = MLPDim(input_dim=out_trend.size(2), output_dim=out_season.size(1)).to(device)

            out_trend_reduced = model(out_trend_flat)

            out_trend_reduced = out_trend_reduced.view(out_trend.size(0), out_trend.size(1), out_season.size(1))
            
            out_trend=out_trend_reduced.permute(0,2,1)

            out = out_season + out_trend
            out = ori + self.out_cross_layer(out)
            # out = ori + self.out_cross_layer(ori)
            out_list.append(out[:, :length, :])
        return out_list

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.daytime=configs.daytime
        self.gapdis=configs.gapdis
        self.use_delay=configs.use_delay
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in
        self.use_future_temporal_feature = configs.use_future_temporal_feature
        self.var_num=configs.var_num

        self.use_kmeans=configs.use_kmeans
        self.use_single_patch=configs.use_single_patch
        self.start_p=[float('inf')]*self.var_num
        self.selected_idx=[]

        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.t_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.channel_independence == 1:
                self.predict_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.seq_len,
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )
            else:
                self.predict_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.t_model,
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )

            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)

                self.out_res_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        configs.seq_len,
                        configs.seq_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ])

                self.regression_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            #configs.seq_len // (configs.down_sampling_window ** i),
                            configs.seq_len,
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )

            self.normalize_layers = torch.nn.ModuleList(
                [
                    Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)#configs.use_norm=1
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

            self.out_process =torch.nn.Linear((configs.down_sampling_layers+1),1)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'avg':
            #down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
            down_pool=nn.ModuleList()
            down_pool.append(torch.nn.AvgPool1d(kernel_size=2, stride=1))
            down_pool.append(torch.nn.AvgPool1d(kernel_size=4, stride=1))
            down_pool.append(torch.nn.AvgPool1d(kernel_size=8, stride=1))            

        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
           
            #print(x_enc_ori.shape)
            dis=2**(i+1)-1
            mean_value=torch.mean(x_enc_ori,dim=2,keepdim=True)
            mean_value_repeated=mean_value.repeat(1,1,dis)
            x_enc_ori=torch.cat((mean_value_repeated,x_enc_ori),dim=2)
            #print(x_enc_ori.shape)
            x_enc_sampling = down_pool[i](x_enc_ori)
            #print(x_enc_sampling.shape)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling
            

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori)
                #x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                #x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, start_p, selected_idx):
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
        
        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')  
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
        
        # embedding
        enc_out_list = []
        #x_list = self.pre_enc(x_list)#单通道不处理；多通道会得到season和trend两个列表
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list)), x_list, x_mark_list):
                #print(x.shape)
                #print(x_mark.shape)
                if self.channel_independence == 1:
                    enc_out = self.enc_embedding(x, x_mark)#[B,T,C]
                else:
                    enc_out = self.enc_embedding(x, x_mark).permute(0,2,1)#[B,T,C]
                #print(f'enc_out{enc_out.shape}')               
                #print(enc_out.shape)
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list)), x_list):
                if self.channel_independence == 1:
                    enc_out = self.enc_embedding(x, None)#[B,T,C]
                else:
                    enc_out = self.enc_embedding(x, None).permute(0,2,1)#[B,T,C]
                enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list, start_p, selected_idx)


        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        #dec_out = torch.mean(torch.stack(dec_out_list, dim=-1),dim=-1)
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        # dec_out=dec_out_list[3]
        #print(dec_out.shape)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        _,_,N=x_list[0].size()
        if self.channel_independence == 1:
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                #print(enc_out.shape)
                if self.use_future_temporal_feature:
                    enc_out = enc_out + self.x_mark_dec
                    enc_out = self.projection_layer(enc_out)
                else:
                    enc_out = self.projection_layer(enc_out)
                
                enc_out = enc_out.reshape(B, self.configs.c_out, self.seq_len).permute(0, 2, 1).contiguous()

                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1) # align temporal dimension
                
                #dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)[:,:,:N]  # align temporal dimension
                #dec_out = self.projection_layer(dec_out)
                dec_out_list.append(dec_out)

        return dec_out_list
    
    def start_singlep_fix(self, x_enc):
        self.use_kmeans=False

        start_p=[]
        start_pk=[float('inf')]*self.var_num
        variable_data=x_enc.permute(2,0,1)

        #需要的输入：torch.Size([7, 256, 96, 16])
        clustered_variables = self.select_fix_variable(variable_data)
        
        x_days=[] #patch  torch.Size([1792, 16, 24])  orch.Size([7, 256, 16, 24])
          
        x_day=variable_data[:,:,0:self.daytime]

        selected_idx=[]
        dim_patch=int(self.daytime)

        for cluster_id, variables in clustered_variables.items():
            # The selected variable for this cluster (fixed variable)
            selected_variable = variables[0]  # First element as the selected_idx, can change if needed
            selected_idx.append(selected_variable)

            x_day_reshape = x_day

            fixed_variable = x_day_reshape[selected_variable, :, 0:self.gapdis]

            for k in variables:
                if k != selected_variable:
                    variable_p = x_day_reshape[k, :, :]

                    best_sim = 0  
                    best_d = 0  
                
                    # Iterate over time steps (d) to find best correlation
                    for d in range(0, dim_patch - self.gapdis):
                        sampled_variable_p = variable_p[:, d:d + self.gapdis]

                        sampled_variable_mean = sampled_variable_p.mean(dim=0, keepdim=True)

                        fixed_variable_mean = fixed_variable.mean(dim=0, keepdim=True)

                        # 计算余弦相似度
                        cos_sim = abs(F.cosine_similarity(sampled_variable_mean, fixed_variable_mean))

                        # Update best correlation and time shift
                        if cos_sim > best_sim:
                            best_sim = cos_sim
                            best_d = d

                    # Store the best time shift for this variable
                    start_pk[k] = best_d

        for i in range():
            start_p.append(start_pk)
        
        if self.use_delay!=True:
            start_p=[]
            for day_index, x_day in enumerate(x_days):
                start_pk=[0]*self.var_num
                start_p.append(start_pk)
        return start_p, selected_idx

    
    def start_p_fix(self, x_enc):

        self.use_kmeans=False

        start_p=[]
        variable_data=x_enc.permute(2,0,1)

        clustered_variables = self.select_fix_variable(variable_data)
        
        x_days=[] #patch  torch.Size([1792, 16, 24])  orch.Size([7, 256, 24])

        x_days=[]
        for i in range(self.seq_len//self.daytime):
            x_days.append(variable_data[:,:,i*self.daytime:(i+1)*self.daytime])
          
        # x_day=variable_data[:,:,0:self.daytime]

        selected_idx=[]
        for cluster_id, variables in clustered_variables.items():
            # The selected variable for this cluster (fixed variable)
            selected_variable = variables[0]  # First element as the selected_idx, can change if needed
            selected_idx.append(selected_variable)
       
        dim_patch=int(self.daytime)
        
        for day_index, x_day in enumerate(x_days):
            start_pk=[float('inf')]*self.var_num

            for cluster_id, variables in clustered_variables.items():
                x_day_reshape = x_day
                selected_variable = variables[0]
                fixed_variable = x_day_reshape[selected_variable, :, int((dim_patch/2-self.gapdis/2)):int((dim_patch/2+self.gapdis/2))]

                for k in variables:
                    if k != selected_variable:
                        variable_p = x_day_reshape[k, :, :]

                        best_sim = 0  
                        best_d = 0  
                
                        # Iterate over time steps (d) to find best correlation
                        for d in range(0, dim_patch-self.gapdis):
                            sampled_variable_p = variable_p[:, d:d + self.gapdis]

                            sampled_variable_mean = sampled_variable_p.mean(dim=0, keepdim=True)

                            fixed_variable_mean = fixed_variable.mean(dim=0, keepdim=True)

                            # 计算余弦相似度
                            cos_sim = abs(F.cosine_similarity(sampled_variable_mean, fixed_variable_mean))

                            # Update best correlation and time shift
                            if cos_sim > best_sim:
                                best_sim = cos_sim
                                best_d = d
                            # Store the best time shift for this variable
                            start_pk[k] = best_d
            start_p.append(start_pk)
        if self.use_delay!=True:
            start_p=[]
            for day_index, x_day in enumerate(x_days):
                start_pk=[0]*self.var_num
                start_p.append(start_pk)

        return start_p, selected_idx


    def select_fix_variable(self, data):

        data_final = data.mean(dim=1)  
        
        print(data_final.shape)
        max_cluster=data.size(0)
        data_reshaped = data_final.cpu().detach().numpy()

        pca = PCA(n_components=1)  
        pca.fit(data_reshaped)
        data_pca = pca.transform(data_reshaped)
        

        range_n_clusters = range(2, 6) 

        best_score = -1  
        best_k = 0 
        best_labels =[]

        for n_clusters in range_n_clusters:
            # KMeans聚类
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(data_pca)

            labels = kmeans.labels_
    
            # 计算轮廓系数
            score = silhouette_score(data_pca, labels)

            if score > best_score:
                best_score = score
                best_k = n_clusters
                best_labels = labels

        clustered_variables = {} 

        for var_index, label in enumerate(best_labels):
            if label not in clustered_variables:
                clustered_variables[label] = []
            clustered_variables[label].append(var_index)

        for cluster_id, variables in clustered_variables.items():
            cluster_center = kmeans.cluster_centers_[cluster_id]
        
            distances = np.linalg.norm(data_pca[variables] - cluster_center, axis=1)
            center_variable_index = variables[np.argmin(distances)]  

            variables.remove(center_variable_index)
            variables.insert(0, center_variable_index)

        
        return clustered_variables
    

    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':

            if self.use_kmeans==True:
                if self.use_single_patch==True:
                    self.start_p,self.selected_idx = self.start_singlep_fix(x_enc)
                else:
                    self.start_p,self.selected_idx = self.start_p_fix(x_enc)
                print(self.start_p)
            

            
            dec_out_list = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, self.start_p, self.selected_idx)
            return dec_out_list
        else:
            raise ValueError('Only forecast tasks implemented yet')

import torch
import torch.nn as nn

class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        #ndim为张量的维数，此处x.ndim=3
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()#dim2reduce其实就是1，也就是在时间上归一化
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

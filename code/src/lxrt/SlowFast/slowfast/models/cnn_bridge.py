
import numpy as np
import torch
import torch.nn as nn
from slowfast.models.temporal_attention import Temporal_Transformer
import torch 
import torch.nn.functional as F
from torch import nn
import numpy as np 
import math 
import torchvision
from torch.autograd import Variable

        
class cnn_bridge_network(nn.Module):
    def __init__(self, hidden_channels, concat_size,attention=False, num_head = 4, query_size = 1000, num_frames = 32 ):
        super().__init__()
        intermediate_hc = int(hidden_channels/2)
        self.fast_avg_pool_3d = nn.AvgPool3d((4, 1, 1), stride=(4, 1, 1))
        self.cnn1 = nn.Conv3d( concat_size , intermediate_hc , (3,3,3) , (1,1,1) , padding=0)
        self.cnn2 = nn.Conv3d( intermediate_hc, hidden_channels , (3,2,2) , (1,2,2) , padding=0)
        self.num_head = num_head
        self.query_size = query_size
        self.num_frames = num_frames
        self.attention = attention
        self.temporal_attention =  Temporal_Transformer(256, num_frames = num_frames, num_features = 256, query_size = query_size, num_head = num_head)
        
    def pool_with_attention(self,x, query):
        batch_size = x.size(0)
        r = self.temporal_attention(x, query)
        attention_w = r[0].contiguous().view(batch_size,self.num_frames,-1)
        x = x.permute(0,1,3,4,2)
        b = x.size(0)
        f = x.size(1)
        h = x.size(2)
        wi = x.size(3)
        x = x.contiguous().view(x.size(0), -1, x.size(-1))
        x = torch.matmul(x,attention_w )
        x = x.contiguous().view(b, f, h, wi, -1)
        return x
    
    def forward(self, x, query=None):
        slow_pathway = x[0]
        fast_pathway = x[1]
        #query = torch.ones(32,20,768)
        if self.attention:
            fast_pathway_pooled = self.pool_with_attention(fast_pathway, query)
        else:
            fast_pathway_pooled = self.fast_avg_pool_3d(fast_pathway)
        concat = torch.cat((fast_pathway_pooled, slow_pathway),1)
        cnn_out1 = self.cnn1(concat)
        cnn_out2 = self.cnn2(cnn_out1)
        bs = int(cnn_out2.size(0))
        feat =  int(cnn_out2.size(1))
        shape = (bs, feat,-1)
        resized_op = torch.reshape(cnn_out2, shape).permute(0,2,1)
        return resized_op
        
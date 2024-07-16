import torch
import torch.nn as nn
import torch.nn.functional as F

class EVFlowNet(nn.Module):
    def __init__(self, args):
        super(EVFlowNet, self).__init__()
        self.no_batch_norm = args.no_batch_norm
        
        # エンコーダー
        self.conv1 = self.conv_layer(4, 64, kernel_size=5, stride=2, padding=2)
        self.conv2 = self.conv_layer(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = self.conv_layer(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv4 = self.conv_layer(256, 256, kernel_size=5, stride=2, padding=2)
        
        # デコーダー
        self.deconv1 = self.deconv_layer(256, 256)
        self.deconv2 = self.deconv_layer(512, 128)
        self.deconv3 = self.deconv_layer(256, 64)
        self.deconv4 = self.deconv_layer(128, 32)
        
        # フロー予測
        self.predict_flow4 = self.predict_flow(256)
        self.predict_flow3 = self.predict_flow(128)
        self.predict_flow2 = self.predict_flow(64)
        self.predict_flow1 = self.predict_flow(32)

    def conv_layer(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1):
        if self.no_batch_norm:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True)
            )

    def deconv_layer(self, in_planes, out_planes):
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def predict_flow(self, in_planes):
        return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)

    
    def forward(self, x):
        # エンコーダー
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)

        # デコーダー
        out_deconv1 = self.deconv1(out_conv4)
        concat1 = torch.cat((out_deconv1, out_conv3), 1)
        out_deconv2 = self.deconv2(concat1)
        concat2 = torch.cat((out_deconv2, out_conv2), 1)
        out_deconv3 = self.deconv3(concat2)
        concat3 = torch.cat((out_deconv3, out_conv1), 1)
        out_deconv4 = self.deconv4(concat3)

        # マルチスケールフロー予測
        flow4 = self.predict_flow4(out_deconv1)
        flow3 = self.predict_flow3(out_deconv2)
        flow2 = self.predict_flow2(out_deconv3)
        flow1 = self.predict_flow1(out_deconv4)

        return [flow1, flow2, flow3, flow4]
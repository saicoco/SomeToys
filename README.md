# _practice
### caffe models prototxt about multimodal
1.关于batchNormalization在caffe中的实现，分为两部分：BatchNorm和Scale两部分,使用时将将conv或Innerproduct中
bias置于Scale中，BatchNorm对无偏置权重乘积进行norm，在Scale中进行放缩并加入bias的学习。  

2.BN的作用很明显，未加入之前loss下降较为缓慢，波动较大；加入之后，audio_loss开始大幅度下降，对最终网络的学习
贡献度更大。

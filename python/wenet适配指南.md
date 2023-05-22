# WeNet适配指南

## wenet流程图(非流式)

![](wenet_pipeline.png)

## 导出onnx
```
# 需提前下载pt模型权重
git clone https://github.com/wenet-e2e/wenet.git
cd wenet
git checkout v2.1.0
cd wenet/bin
python3 export_onnx_gpu.py --config {model_path}/train.yaml --checkpoint {model_path}/final.pt --output_onnx_dir {model_path}/onnx/
```
## 优化onnx并固定shape
```
python3 -m onnxsim encoder.onnx encoder_sim.onnx --input-shape speech:1,2000,80 speech_lengths:1

python3 -m onnxsim decoder.onnx decoder_sim.onnx --input-shape encoder_out:1,499,512 encoder_out_lens:1 hyps_pad_sos_eos:1,10,200 hyps_lens_sos:1,10 r_hyps_pad_sos_eos:1,10,200 ctc_score:1,10
```


## 转换OM模型
```
atc --model encoder_sim.onnx --output encoder_static --framework=5 --soc_version=Ascend310P3
atc --model decoder_sim.onnx --output decoder_static --framework=5 --soc_version=Ascend310P3
```

## 运行Demo
安装依赖
```
pip install soundfile
pip install kaldifeat
pip install torch
```
修改wenet.py中以下字段
```
args['encoder_om_path'] = '/home/yolaw/wenet/models/om/encoder_static.om' # encoder模型路径
args['decoder_om_path'] = '/home/yolaw/wenet/models/om/decoder_static.om' # decoder模型路径
waveform, sample_rate = sf.read("/home/yolaw/test3.wav") # 处理语音文件
```

执行demo
```
python3 wenet.py
```
得到运行结果
```
# python3 wenet.py
input node[0] info:{'name': 'speech', 'dimCount': 3, 'dims': [1, 2000, 80]}
input node[1] info:{'name': 'speech_lengths', 'dimCount': 1, 'dims': [1]}
output node[0] info:{'name': 'Add_2447:0:encoder_out', 'dimCount': 3, 'dims': [1, 499, 512]}
output node[1] info:{'name': 'Cast_2456:0:encoder_out_lens', 'dimCount': 1, 'dims': [1]}
output node[2] info:{'name': 'LogSoftmax_2455:0:ctc_log_probs', 'dimCount': 3, 'dims': [1, 499, 5538]}
output node[3] info:{'name': 'TopK_2457:0:beam_log_probs', 'dimCount': 3, 'dims': [1, 499, 10]}
output node[4] info:{'name': 'TopK_2457:1:beam_log_probs_idx', 'dimCount': 3, 'dims': [1, 499, 10]}
init resource success
input node[0] info:{'name': 'encoder_out', 'dimCount': 3, 'dims': [1, 499, 512]}
input node[1] info:{'name': 'encoder_out_lens', 'dimCount': 1, 'dims': [1]}
input node[2] info:{'name': 'hyps_pad_sos_eos', 'dimCount': 3, 'dims': [1, 10, 200]}
input node[3] info:{'name': 'hyps_lens_sos', 'dimCount': 2, 'dims': [1, 10]}
input node[4] info:{'name': 'r_hyps_pad_sos_eos', 'dimCount': 3, 'dims': [1, 10, 200]}
input node[5] info:{'name': 'ctc_score', 'dimCount': 2, 'dims': [1, 10]}
output node[0] info:{'name': 'PartitionedCall_ArgMax_1349_ArgMaxV2_9:0:best_index', 'dimCount': 1, 'dims': [1]}
init resource success
data interaction from host to device
data interaction from host to device success
model inference success
data interaction from device to host
data interaction from device to host success
encoder_out_len: 285
data interaction from host to device
data interaction from host to device success
model inference success
data interaction from device to host
data interaction from device to host success
('最容易想到的策略是贪心搜索及每一个时间部都取出一个条件概率最大的输出再将从开始到当节目的结果作为输入去获得下一个时间部的输出直到模型给出生成结束的标志',)

```
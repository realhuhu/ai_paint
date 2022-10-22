## 使用说明

### 导入

```python
from utils import load_novelAI,KDiffusionSampler,txt2img,img2img,FaceRestorerCodeFormer
```

#### 1.novelAI模型

```python
sd_model=load_novelAI(checkpoint_file,vae_file,hypernetwork_file,config,CLIP_stop)
```

checkpoint_file：ckpt文件的路径

vae_file：.vae.pt文件的路径

hypernetwork_file:hypernetwork文件的路径，不想使用hypernetwork就设为None

config：config文件的路径，默认为"./config.yaml"

CLIP_stop：CLIP_stop的层，默认为2

这一步会有warning，不用管

#### 2.Sampler

```python
sampler=KDiffusionSampler("sample_euler_ancestral",sd_model)
```

sample_euler_ancestral：sampler名称，默认用它就行了

sd_model：之前生成的novelAI模型

#### 3.人脸增强

```python
face_restoration=FaceRestorerCodeFormer(face_restoration_file)
```

目前人脸修复可能影响清晰度，酌情使用

face_restoration_file：codeformer.pth文件的位置，建议放在./weights/facelib/codeformer.pth，[下载链接](https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth)

使用人脸修复还需要下载两个模型，会自动下载，你也可以根据提示手动下载，保存到weights/facelib文件夹下，注意名称要和图中一致

[detection_Resnet50_Final.pth](https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth)

[parsing_parsenet](https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth)

![image-20221021165929910](https://typora-1304907527.cos.ap-nanjing.myqcloud.com/202210211659928.png)

第一次使用会有warning，不用管

### 使用

#### 1.文转图

```python
txt2img(
        sd_model,# novelAI模型
        sampler,# Sampler
        prompt,# 想要的特征，字符串
        negative_prompt,# 不想要的特征，字符串
        *,
        height=512,# 图片高度
        width=512,# 图片宽度
        steps=28,# 迭代次数
        cfg=7,# 精细度
        eta=0.21,# eta
        seed=None,# 种子，一样的话生成结果基本相同
        subseed=None,# 副种子，轻微影响结果
        face_restoration: FaceRestorerCodeFormer = None,# 人脸修复模型
        face_restoration_weight=1# 人脸修复强度
)
```

如

![image-20221021165209592](https://typora-1304907527.cos.ap-nanjing.myqcloud.com/202210211652693.png)

#### 2.图转图

```python
txt2img(
        sd_model,# novelAI模型
        sampler,# Sampler
    	img,# 原图，类型为PIL.Image
        prompt,# 想要的特征，字符串
        negative_prompt,# 不想要的特征，字符串
        *,
    	resize_mode=1,# 图片大小调整模式，1直接拉伸、2放大填满、3放缩填充（自动补齐空缺部分）
        height=512,# 成图高度
        width=512,# 成图宽度
        steps=28,# 迭代次数
        cfg=7,# 精细度
    	denoising_strength=0.75,# 与原图差异度
        eta=0.21,# eta
        seed=None,# 种子，一样的话生成结果基本相同
        subseed=None,# 副种子，轻微影响结果
        face_restoration: FaceRestorerCodeFormer = None,# 人脸修复模型
        face_restoration_weight=1# 人脸修复强度
)
```

如

![image-20221021165743567](https://typora-1304907527.cos.ap-nanjing.myqcloud.com/202210211657655.png)
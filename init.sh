#创建虚拟环境
conda create -n qwen2.5vl python=3.12 -y  
conda activate qwen2.5vl
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

#拉取LLaMA-factory仓库代码，安装依赖
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
pip install swanlab unzip
sudo -v ; curl https://gosspublic.alicdn.com/ossutil/install.sh | sudo bash
#如有问题，可参照LLaMA-Factory源码安装环境, 源码链接：https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md?plain=1#L471

#安装Qwen2.5-VL
git clone https://github.com/QwenLM/Qwen2.5-VL.git
cd Qwen2.5-VL
pip install qwen-vl-utils[decord]
pip install transformers
pip install 'accelerate>=0.26.0'
#如有问题，可参照Qwen2.5-VL源码安装环境，源码链接：https://github.com/QwenLM/Qwen2.5-VL

#当前镜像已经创建好虚拟环境，使用conda info --envs查看，激活使用即可

#下面是从魔搭社区拉取Qwen2.5-VL-7B-Instruct模型，如有问题，可参照魔搭社区模型下载文档, 链接：https://modelscope.cn/docs/models/download
pip install modelscope
modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct --local_dir /root/autodl-tmp/Qwen2.5-VL-7B-Instruct
#!/bin/sh
set -ex

# wget -O audio2face/ds_graph/output_graph.pb https://deep-learning-1253526705.cos.ap-beijing.myqcloud.com/facial/output_graph.pb

# wget -O video_preprocess.zip https://deep-learning-1253526705.cos.ap-beijing.myqcloud.com/facial/video_preprocess.zip
# unzip -o video_preprocess.zip

# cd face_render/face3d/mesh/cython
# python3 face_render/face3d/mesh/cython/setup.py build_ext -i
# cd ../../../

mkdir -p face_render/BFM
wget -O face_render/BFM/BFM_model_front.mat https://deep-learning-1253526705.cos.ap-beijing.myqcloud.com/facial/BFM_model_front.mat
wget -O face_render/BFM/std_exp.txt https://deep-learning-1253526705.cos.ap-beijing.myqcloud.com/facial/std_exp.txt

# 替换掉python文件里的绝对路径
sed -i 's/\/content\/FACIAL/../g' face_render/*.py

python face_render/handle_netface.py
python face_render/fit_headpose.py
python face_render/render_netface_fitpose.py

# cd ..
ffmpeg -i video_preprocess/train1.mp4 -y -acodec pcm_s16le -f wav -ac 1 -ar 16000  video_preprocess/train1.wav

cp -r video_preprocess/train1.wav examples/audio/train1.wav
cp -r video_preprocess/test1.wav examples/audio/test1.wav

# cd audio2face
python audio2face/audio_preprocessing.py

mkdir -p audio2face/checkpoint/obama
mkdir -p audio2face/data
wget -O audio2face/checkpoint/obama/Gen-20-0.0006273046686902202.mdl https://deep-learning-1253526705.cos.ap-beijing.myqcloud.com/facial/Gen-20-0.0006273046686902202.mdl
wget -O audio2face/data/train3.npz https://deep-learning-1253526705.cos.ap-beijing.myqcloud.com/facial/train3.npz

# 替换掉python文件里的绝对路径
sed -i 's/\/content\/FACIAL/../g' audio2face/*.py
python audio2face/fintuning2-trainheadpose.py

#"""## 3.2. Test audio2face"""
python audio2face/test.py --audiopath examples/audio_preprocessed/test1.pkl --checkpath audio2face/checkpoint/train1/Gen-10.mdl

#"""## 4.1. Run 3D face rendering"""
# cd ../face_render/
python3 face_redner/rendering_gaosi.py --train_params_path video_preprocess/train1_posenew.npz --net_params_path examples/test-result/test1.npz

#"""## 4.2. Prepare traning and testing data"""
# cd ..
cp -r video_preprocess/train_A face2vid/datasets/train3/train_A
cp -r video_preprocess/train1_image face2vid/datasets/train3/train_B

rm -rf face2vid/datasets/train3/test_A /face2vid/datasets/train3/test_B
cp -r examples/rendering/test1 face2vid/datasets/train3/test_A
cp -r examples/rendering/test1 face2vid/datasets/train3/test_B

#"""##5.1.1 Download checkpoint for face2video (optional 1)"""

# cd face2vid
mkdir -p face2vid/checkpoints/train3/
wget -O face2vid/checkpoints/train3/latest_net_G.pth https://deep-learning-1253526705.cos.ap-beijing.myqcloud.com/facial/50_net_G.pth

#"""##5.1.2 Train face2video by yourself (optional 2)"""

# Commented out IPython magic to ensure Python compatibility.
### Please pay attention to modify blink_path when you train your own video. 
### Here we use 7200 images for training. (Usually 4000 images are enough.) 
python3 face2vid/train.py --blink_path video_preprocess/train1_openface/train1_512_audio.csv --name train3 --model pose2vid --dataroot face2vid/datasets/train3/ --netG local --ngf 32 --num_D 3 --tf_log --niter_fix_global 0 --label_nc 0 --no_instance --save_epoch_freq 2 --lr=0.0001 --resize_or_crop resize --no_flip --verbose --n_local_enhancers 1


#"""## 5.2 Test face2video"""

python3 face2vid/test_video.py --test_id_name test1 --blink_path examples/test-result/test1.npz --name train3 --model pose2vid --dataroot face2vid/datasets/train3/ --which_epoch latest --netG local --ngf 32 --label_nc 0 --n_local_enhancers 1 --no_instance --resize_or_crop resize


audio_path ='video_preprocess/test1.wav'
video_new='examples/test_image/test1/test_1.avi'
output = 'examples/test_image/test1/test_1_audio.avi'
output_mp4 = 'examples/test_image/test1/test_1_audio.mp4'
ffmpeg -i "$video_new" -i "$audio_path" -c copy "$output"
ffmpeg -i "$output"  "$output_mp4"


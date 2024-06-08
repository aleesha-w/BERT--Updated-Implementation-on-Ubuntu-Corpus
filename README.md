
# BERT- Updated Implementation on Ubuntu Corpus

## Requirements
Colab TPU \
Google Bucket (This is where the checkpoints will be stored)

## Download the tfrecord.zip file
https://drive.google.com/file/d/15PlJFy4BbGag4QACZ1-gGRcEe6SLveb8/view?usp=sharing \
This file contains the preprocessed data from the Ubuntu Corpus dataset. You may also preprocess it yourself with the available code files without the use of a TPU.

## Setting up the Environment 
1. Unzip the tfrecord file \
   `!unzip tfrecord.zip`
2. Identify the TPU address \
   `import tensorflow as tf
   try:
	tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  
	print('Running on TPU ', tpu.cluster_spec().as_dict( ['worker'])
   except ValueError:
	raise BaseException('ERROR: Not connected to a TPU runtime')
print(tpu.cluster_spec().as_dict()['worker'])

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.TPUStrategy(tpu)`

3. Connect to your Google Bucket \
   `from google.colab import auth
  auth.authenticate_user()
  project_id = 'xxxx'
  !gcloud config set project {project_id}`

   `bucket_name = 'xxxx' + str(uuid.uuid1())
   !gsutil mb gs://{bucket_name}`

## Training
To begin training the model, use the code below or paste it into a shell file. Be sure to modify the TPU address to match the one obtained from step 2 above. You must also modify the output directory. \
`
!python run_pretraining.py \
    --input_file='gs://tf_record_data_buck67f12a58-6aec-11ee-a9db-0242ac1c000c/tf_train.tfrecord' \
    --output_dir='gs://out_file' \
    --do_train=True \
    --do_eval=True \
    --bert_config_file=/content/bert/config.json \
    --train_batch_size=64 \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --num_train_steps=100 \
    --num_warmup_steps=10 \
    --learning_rate=2e-5 \
    --use_tpu=True \
    --tpu_name=grpc://10.91.24.210:8470`

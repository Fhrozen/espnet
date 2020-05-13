### Docker

```bash
$ ./run.sh --docker_gpu 0 --docker_cuda 10.1 --docker_egs fearless/raw_asr1 --docker_folders /home/nelson/corpus --ngpu 4 --datadir /home/nelson/corpus/FS02_Challenge_Data --stop_stage 3

$ ./run.sh --docker_gpu 0,1,2,3 --docker_cuda 10.1 --docker_egs fearless/raw_asr1 --ngpu 4 --stage 4 --verbose 0 --train_config train/transformer_convutt.yaml
./run.sh --docker_gpu 0 --docker_cuda 10.1 --docker_egs fearless/asr1 --ngpu 0 --stage 5

./run.sh --docker_gpu 0 --docker_cuda 10.1 --docker_egs fearless/raw_asr1 --ngpu 1 --stage 4 --train_config conf/train/transformer_resbnutt.yaml --preprocess_config conf/preprocess/fbank.yaml
```
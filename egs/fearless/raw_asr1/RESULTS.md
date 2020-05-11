### Docker

```bash
$ ./run.sh --docker_gpu 0 --docker_cuda 10.1 --docker_egs fearless/raw_asr1 --docker_folders /home/nelson/corpus --ngpu 4 --datadir /home/nelson/corpus/FS02_Challenge_Data --stop_stage 3

$ ./run.sh --docker_gpu 0,1,2,3 --docker_cuda 10.1 --docker_egs fearless/asr1 --ngpu 4 --stage 4 --verbose 1
./run.sh --docker_gpu 0 --docker_cuda 10.1 --docker_egs fearless/asr1 --ngpu 0 --stage 5
```
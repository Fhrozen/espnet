# PyTorch version: The latest six versions have been tested.
# Please check https://github.com/espnet/espnet?tab=readme-ov-file#espnet-end-to-end-speech-processing-toolkit
TH_VERSION := 2.6.0

# Use pip for pytorch installation even if you have anaconda
ifneq ($(shell test -f ./activate_python.sh && grep 'conda activate' ./activate_python.sh),)
USE_CONDA := 1
else
USE_CONDA :=
endif


# Set if install binaries on CPU mode e.g. make CPU_ONLY=0
# If you don't have nvcc, this value will be set automatically
ifneq ($(shell which nvcc 2>/dev/null),)
CPU_ONLY :=
# Derive CUDA version from nvcc
CUDA_VERSION := $(shell nvcc --version | grep "Cuda compilation tools" | cut -d" " -f5 | sed s/,//)
CUDA_VERSION_WITHOUT_DOT := $(strip $(subst .,,$(CUDA_VERSION)))

else
CPU_ONLY := 0
CUDA_VERSION :=
CUDA_VERSION_WITHOUT_DOT :=
endif
WITH_OMP=ON

.PHONY: all clean

all: showenv conda_packages.done python ffmpeg.done sctk check_install

python: activate_python.sh setuptools.done packaging.done espnet.done pytorch.done chainer.done fairscale.done torch_optimizer.done flash_attn.done lightning.done
extra: warp-transducer.done nkf.done moses.done mwerSegmenter.done pesq kenlm.done pyopenjtalk.done py3mmseg.done beamformit.done fairseq.done s3prl.done k2.done transformers.done phonemizer.done longformer.done muskits.done whisper.done rvad_fast.done sounfile_test parallel-wavegan.done lora.done sph2pipe torcheval.done

activate_python.sh:
	test -f activate_python.sh || { echo "Error: Run ./setup_python.sh or ./setup_miniforge.sh"; exit 1; }

################ Logging ################
showenv: activate_python.sh
ifeq ($(strip $(CPU_ONLY)),)
	@echo CUDA_VERSION=$(CUDA_VERSION)
else
	@echo Perform on CPU mode: CPU_ONLY=$(CPU_ONLY)
endif
	@echo PYTHON=$(shell . ./activate_python.sh && command -v python3)
	@echo PYTHON_VERSION=$(shell . ./activate_python.sh && python3 --version)
	@echo USE_CONDA=$(USE_CONDA)
	@echo TH_VERSION=$(TH_VERSION)
	@echo WITH_OMP=$(WITH_OMP)

#########################################

bc.done: activate_python.sh
	. ./activate_python.sh && { command -v bc || conda install -y bc -c conda-forge; }
	touch bc.done
cmake.done: activate_python.sh
	. ./activate_python.sh && { command -v cmake || conda install -y "cmake<4.0"; }
	touch cmake.done
flac.done: activate_python.sh
	. ./activate_python.sh && { command -v flac || conda install -y libflac -c conda-forge; }
	touch flac.done
sox.done: activate_python.sh
	. ./activate_python.sh && { command -v sox || conda install -y sox -c conda-forge; }
	touch sox.done
sndfile.done: activate_python.sh
	# NOTE(kamo): The wheel version of Soundfile includes shared library and it is refered in preference.
	# However, some old linuxs are not compatible with the wheel, so libsndfile is installed for this case.
	. ./activate_python.sh && { conda install -y libsndfile -c conda-forge; }
	touch sndfile.done
ifneq ($(strip $(USE_CONDA)),)
conda_packages.done: bc.done cmake.done flac.done sox.done sndfile.done
else
conda_packages.done:
endif
	touch conda_packages.done

ffmpeg.done: activate_python.sh
ifneq ($(strip $(USE_CONDA)),)
	. ./activate_python.sh && { command -v ffmpeg || conda install -y ffmpeg -c conda-forge; }
else
	. ./activate_python.sh && { command -v ffmpeg || ./installers/install_ffmpeg.sh; }
endif
	touch ffmpeg.done

sctk: sctk/bin/sclite
sctk/bin/sclite:
	./installers/install_sctk.sh

sph2pipe: sph2pipe/sph2pipe
sph2pipe/sph2pipe:
	./installers/install_sph2pipe.sh

setuptools.done: activate_python.sh
	# NOTE(jiatong): For python 3.12 case, as distutils are not default from 3.12
	. ./activate_python.sh && { python -m ensurepip --upgrade; pip install setuptools; }
	touch setuptools.done

packaging.done: setuptools.done
	. ./activate_python.sh && python3 -m pip install packaging
	touch packaging.done

numpy.done: activate_python.sh
ifeq ($(strip $(USE_CONDA)),)
	. ./activate_python.sh && python3 -m pip install numpy
else
	. ./activate_python.sh && conda install -y numpy
endif
	touch numpy.done

numba.done: numpy.done
	. ./activate_python.sh && python3 -m pip install -U numba
	touch numba.done

# NOTE(kamo): Install numba before installing torch because numba requires specific numpy version now
# (Pytorch is not related to numba directly)
pytorch.done: packaging.done numba.done
ifeq ($(strip $(USE_CONDA)),)
	. ./activate_python.sh && ./installers/install_torch.sh "false" "${TH_VERSION}" "${CUDA_VERSION}"
else
	. ./activate_python.sh && ./installers/install_torch.sh "true" "${TH_VERSION}" "${CUDA_VERSION}"
endif
	touch pytorch.done

lightning.done: pytorch.done
	. ./activate_python.sh && ./installers/install_lightning.sh
	touch lightning.done

# NOTE(kamo): conda_packages is not necessary for installation of espnet, but add it the dependencies just in case.
espnet.done: pytorch.done conda_packages.done lightning.done
	@echo NUMPY_VERSION=$(shell . ./activate_python.sh && python3 -c "import numpy; print(numpy.__version__)")
	. ./activate_python.sh && python3 -m pip install -e "..[train, recipe]"  # Install editable mode by default
	@echo NUMPY_VERSION=$(shell . ./activate_python.sh && python3 -c "import numpy; print(numpy.__version__)")
	. ./activate_python.sh && python -m nltk.downloader averaged_perceptron_tagger_eng
	touch espnet.done

sounfile_test: espnet.done
	. ./activate_python.sh && python3 ./test_soundfile.py

chainer.done: espnet.done
	. ./activate_python.sh && ./installers/install_chainer.sh "${CUDA_VERSION}"
	touch chainer.done

warp-transducer.done: pytorch.done conda_packages.done
ifeq ($(strip $(CPU_ONLY)),)
	[ -n "${CUDA_HOME}" ] || { echo -e "Error: CUDA_HOME is not set.\n    $$ . ./setup_cuda_env.sh <cuda-root>"; exit 1; }
endif
	. ./activate_python.sh && ./installers/install_warp-transducer.sh ${WITH_OMP}
	touch warp-transducer.done

chainer_ctc.done: chainer.done
ifeq ($(strip $(CPU_ONLY)),)
	[ -n "${CUDA_HOME}" ] || { echo -e "Error: CUDA_HOME is not set.\n    $$ . ./setup_cuda_env.sh <cuda-root>"; exit 1; }
endif
	. ./activate_python.sh && ./installers/install_chainer_ctc.sh ${WITH_OMP}
	touch chainer_ctc.done

nkf.done:
	./installers/install_nkf.sh
	touch nkf.done

pyopenjtalk.done: espnet.done conda_packages.done
	. ./activate_python.sh && ./installers/install_pyopenjtalk.sh
	touch pyopenjtalk.done

phonemizer.done: espnet.done conda_packages.done
ifeq ($(WITH_OMP),ON)
	. ./activate_python.sh && ./installers/install_phonemizer.sh
	touch phonemizer.done
else
	# FIXME(kamo): I don't know how to avoid "-fopenmp" option
	echo "Warning: ./installers/install_phonemizer.sh requires WITH_OMP=ON"
endif

speechbrain.done: espnet.done conda_packages.done
	. ./activate_python.sh && ./installers/install_speechbrain.sh
	touch speechbrain.done

mfa.done: espnet.done pyopenjtalk.done phonemizer.done conda_packages.done
ifneq ($(strip $(USE_CONDA)),)
	. ./activate_python.sh && ./installers/install_mfa.sh "true"
else
	. ./activate_python.sh && ./installers/install_mfa.sh "false"
endif
	touch mfa.done

moses.done:
	git clone --depth 1 https://github.com/moses-smt/mosesdecoder.git moses
	touch moses.done

mwerSegmenter.done:
	./installers/install_mwerSegmenter.sh
	touch mwerSegmenter.done

kenlm.done: espnet.done conda_packages.done
	. ./activate_python.sh && ./installers/install_kenlm.sh
	touch kenlm.done

pesq: PESQ/P862_annex_A_2005_CD/source/PESQ
PESQ/P862_annex_A_2005_CD/source/PESQ:
	./installers/install_pesq.sh

py3mmseg.done: espnet.done
	. ./activate_python.sh && ./installers/install_py3mmseg.sh
	touch py3mmseg.done

beamformit.done:
	./installers/install_beamformit.sh
	touch beamformit.done

torch_optimizer.done: espnet.done
	. ./activate_python.sh && ./installers/install_torch_optimizer.sh
	touch torch_optimizer.done

fairscale.done: espnet.done
	. ./activate_python.sh && ./installers/install_fairscale.sh
	touch fairscale.done

fairseq.done: espnet.done
	. ./activate_python.sh && ./installers/install_fairseq.sh
	touch fairseq.done

s3prl.done: espnet.done
	. ./activate_python.sh && ./installers/install_s3prl.sh
	touch s3prl.done

whisper.done: espnet.done
	. ./activate_python.sh && ./installers/install_whisper.sh
	touch whisper.done

lora.done: espnet.done
	. ./activate_python.sh && ./installers/install_lora.sh
	touch lora.done

k2.done: espnet.done
	. ./activate_python.sh && ./installers/install_k2.sh
	touch k2.done

gtn.done: espnet.done
	. ./activate_python.sh && ./installers/install_gtn.sh
	touch gtn.done

transformers.done: espnet.done
	. ./activate_python.sh && ./installers/install_transformers.sh
	touch transformers.done

torcheval.done: pytorch.done
	. ./activate_python.sh && ./installers/install_torcheval.sh
	touch torcheval.done

parallel-wavegan.done: espnet.done
	. ./activate_python.sh && ./installers/install_parallel-wavegan.sh
	touch parallel-wavegan.done

longformer.done: espnet.done
	. ./activate_python.sh && ./installers/install_longformer.sh
	touch longformer.done

flash_attn.done: espnet.done
	. ./activate_python.sh && ./installers/install_flash_attn.sh
	touch flash_attn.done

muskits.done: espnet.done
	. ./activate_python.sh && ./installers/install_muskits.sh
	touch muskits.done

cauchy_mult.done: espnet.done
	. ./activate_python.sh && ./installers/install_cauchy_mult.sh
	touch cauchy_mult.done

rvad_fast.done: espnet.done
	git clone https://github.com/zhenghuatan/rVADfast.git
	touch rvad_fast.done

discrete_speech_metrics.done: espnet.done
	. ./activate_python.sh && ./installers/install_discrete_speech_metrics.sh
	touch discrete_speech_metrics.done

versa.done: espnet.done
	. ./activate_python.sh && ./installers/install_versa.sh
	touch versa.done

check_install: python
	. ./activate_python.sh; . ./extra_path.sh; python3 check_install.py

clean: clean_extra
	rm -rf warp-transducer chainer_ctc
	rm -rf *.done
	find . -iname "*.pyc" -delete

clean_python:
	rm -rf warp-transducer chainer_ctc
	rm -f chainer_ctc.done espnet.done chainer.done pytorch.done warp-transducer.done
	find . -iname "*.pyc" -delete

clean_extra:
	rm -rf nkf.done swig.done moses.done mwerSegmenter.done
	rm -rf hts_engine_API.done open_jtalk.done pyopenjtalk.done
	rm -rf muskits.done
	rm -rf rvad_fast.done
	rm -rf lightning_constraints.txt
	rm -rf chainer espeak-ng festival MBROLA ParallelWaveGAN versa
	rm -rf py3mmseg sctk* speech_tools sph2pipe* ._mwerSegmenter
	rm -rf nkf mecab swig moses mwerSegmenter
	rm -rf PESQ PESQ.zip

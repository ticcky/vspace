#THEANO_FLAGS="mode=FAST_RUN,floatX=float32,device=cpu" strace -f python experiment_DataSize.py --nworkers $1 --init_b --out_tag $2 --debug
THEANO_FLAGS="mode=FAST_RUN,floatX=float32,device=cpu" python experiment_DataSize.py --nworkers $1 --init_b --out_tag $2 --debug
rm -rf /tmp/tmp_vspace*

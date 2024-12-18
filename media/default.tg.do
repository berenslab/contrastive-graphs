redo-ifchange $2

# import certifi; certifi.where()
# SSL file in the singularity container
export SSL_CERT_FILE=/usr/local/lib/python3.12/dist-packages/certifi/cacert.pem

PROJROOT=$(dirname $PWD)
SINGULARITYFLAGS="--pwd $PWD --bind $PROJROOT,$XDG_CACHE_DIR --env PYTHONPATH=$PROJROOT/src"
RUN="singularity exec $SINGULARITYFLAGS ../nik.sif python3"
$RUN telegram_send.py $2

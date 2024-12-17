redo-ifchange $2
redo-ifchange ../.RUN
RUN=$(<../.RUN)
$RUN telegram_send.py $2

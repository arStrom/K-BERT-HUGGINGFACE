CUDA_VISIBLE_DEVICES='0' python3 -u run_kbert_transformers_cls.py --model ernie-rnn --pretrained ernie2 --epochs_num 20 --batch_size 12 --seq_length 512 --kg_name CnDbpedia --cuda --no_kg > ./outputs/kbert_multibookclassification_erniercnn_ernie2_adaw_no_kg_10040157.log 2>&1 &
CUDA_VISIBLE_DEVICES='1' python3 -u run_kbert_transformers_cls.py --model ernie-rnn --pretrained ernie2 --epochs_num 20 --batch_size 12 --seq_length 512 --kg_name CnDbpedia --cuda > ./outputs/kbert_multibookclassification_erniernn_ernie2_adaw_10040157.log 2>&1 &
wait
echo "ernie-rnn ernie2执行完毕"
CUDA_VISIBLE_DEVICES='0' python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --epochs_num 20 --batch_size 12 --seq_length 512 --kg_name CnDbpedia --cuda --no_kg > ./outputs/kbert_multibookclassification_erniercnn_ernie1_adaw_no_kg_10040157.log 2>&1 & 
CUDA_VISIBLE_DEVICES='1' python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie2 --epochs_num 20 --batch_size 12 --seq_length 512 --kg_name CnDbpedia --cuda --no_kg > ./outputs/kbert_multibookclassification_erniercnn_ernie2_adaw_no_kg_10040157.log 2>&1 &
wait
echo "ernie-rcnn ernie1 和ernir2 执行完毕"
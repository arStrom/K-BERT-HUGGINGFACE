python3 -u run_kbert_transformers_cls.py --model ernie --pretrained ernie3 --epochs_num 20 --batch_size 32 --seq_length 512 --kg_name CnDbpedia --cuda --no_kg > ./outputs/kbert_multibookclassification_ernie_ernie3_adaw_no_kg_10040157.log 2>&1 &
wait
echo "ernie ernie3 no kg执行完毕"
python3 -u run_kbert_transformers_cls.py --model ernie --pretrained ernie3 --epochs_num 20 --batch_size 32 --seq_length 512 --kg_name CnDbpedia --cuda > ./outputs/kbert_multibookclassification_ernie_ernie3_adaw_10040157.log 2>&1 &
wait
echo "ernie ernie3 kg执行完毕"
python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie3 --epochs_num 20 --batch_size 32 --seq_length 512 --kg_name CnDbpedia --cuda --no_kg > ./outputs/kbert_multibookclassification_erniercnn_ernie3_adaw_no_kg_10040157.log 2>&1 &
wait
echo "ernie-rcnn ernie3 no kg执行完毕"
python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie3 --epochs_num 20 --batch_size 32 --seq_length 512 --kg_name CnDbpedia --cuda > ./outputs/kbert_multibookclassification_erniercnn_ernie3_adaw_10040157.log 2>&1 &
wait
echo "ernie-rcnn ernie3 kg执行完毕"
python3 -u run_kbert_transformers_cls.py --model ernie-rnn --pretrained ernie3 --epochs_num 20 --batch_size 32 --seq_length 512 --kg_name CnDbpedia --cuda --no_kg > ./outputs/kbert_multibookclassification_erniernn_ernie3_adaw_no_kg_10040157.log 2>&1 &
wait
echo "ernie-rnn ernie3 no kg执行完毕"
python3 -u run_kbert_transformers_cls.py --model ernie-rnn --pretrained ernie3 --epochs_num 20 --batch_size 32 --seq_length 512 --kg_name CnDbpedia --cuda > ./outputs/kbert_multibookclassification_erniernn_ernie3_adaw_10040157.log 2>&1 &
wait
echo "ernie-rnn ernie3执行完毕"
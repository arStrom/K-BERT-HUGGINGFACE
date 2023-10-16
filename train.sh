# CUDA_VISIBLE_DEVICES='0' python3 -u run_kbert_transformers_cls.py --model ernie-rnn --pretrained ernie2 --epochs_num 20 --batch_size 12 --seq_length 512 --kg_name CnDbpedia --cuda --no_kg > ./outputs/kbert_multibookclassification_erniercnn_ernie2_adaw_no_kg_10040157.log 2>&1 &
# CUDA_VISIBLE_DEVICES='1' python3 -u run_kbert_transformers_cls.py --model ernie-rnn --pretrained ernie2 --epochs_num 20 --batch_size 12 --seq_length 512 --kg_name CnDbpedia --cuda > ./outputs/kbert_multibookclassification_erniernn_ernie2_adaw_10040157.log 2>&1 &
# wait
# echo "ernie-rnn ernie2执行完毕"
# CUDA_VISIBLE_DEVICES='0' python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --epochs_num 20 --batch_size 12 --seq_length 512 --kg_name CnDbpedia --cuda --no_kg > ./outputs/kbert_multibookclassification_erniercnn_ernie1_adaw_no_kg_10040157.log 2>&1 & 
# CUDA_VISIBLE_DEVICES='1' python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie2 --epochs_num 20 --batch_size 12 --seq_length 512 --kg_name CnDbpedia --cuda --no_kg > ./outputs/kbert_multibookclassification_erniercnn_ernie2_adaw_no_kg_10040157.log 2>&1 &
# wait
# echo "ernie-rcnn ernie1 和ernir2 执行完毕"

CUDA_VISIBLE_DEVICES='1' nohup python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC-slice --dataset book_multilabels_task_slice --seq_length 256 --epochs_num 40 --batch_size 8 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC_slice-256-0.1-0.1-40-8-2.00E_05-100-CnDbpedia-FALSE-FALSE-AdamW.log 2>&1 &
CUDA_VISIBLE_DEVICES='0' nohup python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC-slice --dataset book_multilabels_task_slice --seq_length 256 --epochs_num 40 --batch_size 8 --kg_name CnDbpedia --no_kg --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC_slice-256-0.1-0.1-40-8-2.00E_05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
wait
echo "4号5号执行完毕"
CUDA_VISIBLE_DEVICES='1' nohup python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC-slice --dataset book_multilabels_task_slice --learning_rate 1e-4 --seq_length 256 --epochs_num 20 --batch_size 8 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC_slice-256-0.1-0.1-20-8-1.00E-04-100-CnDbpedia-FALSE-FALSE-AdamW.log 2>&1 &
CUDA_VISIBLE_DEVICES='0' nohup python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC-slice --dataset book_multilabels_task_slice --learning_rate 1e-4 --seq_length 256 --epochs_num 20 --batch_size 8 --kg_name CnDbpedia --no_kg --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task_slice-MLC_slice-256-0.1-0.1-20-8-1.00E-04-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
wait
echo "6号7号执行完毕"
CUDA_VISIBLE_DEVICES='1' nohup python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 2e-5 --seq_length 256 --epochs_num 20 --batch_size 8 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task-MLC-256-0.1-0.1-20-8-2.00E-05-100-CnDbpedia-FALSE-FALSE-AdamW.log 2>&1 &
CUDA_VISIBLE_DEVICES='0' nohup python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 2e-5 --seq_length 256 --epochs_num 20 --batch_size 8 --kg_name CnDbpedia --no_kg --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task-MLC-256-0.1-0.1-20-8-2.00E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
wait
echo "8号9号执行完毕"
CUDA_VISIBLE_DEVICES='1' nohup python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 2e-5 --seq_length 256 --epochs_num 40 --batch_size 8 --kg_name CnDbpedia --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task-MLC-256-0.1-0.1-40-8-2.00E-05-100-CnDbpedia-FALSE-FALSE-AdamW.log 2>&1 &
CUDA_VISIBLE_DEVICES='0' nohup python3 -u run_kbert_transformers_cls.py --model ernie-rcnn --pretrained ernie1 --task MLC --dataset book_multilabels_task --learning_rate 2e-5 --seq_length 256 --epochs_num 40 --batch_size 8 --kg_name CnDbpedia --no_kg --cuda > ./outputs/ernie-rcnn-ernie1-book_multilabels_task-MLC-256-0.1-0.1-40-8-2.00E-05-100-CnDbpedia-TRUE-TRUE-AdamW.log 2>&1 &
wait
echo "10号11号执行完毕"